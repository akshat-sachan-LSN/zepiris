# EC2 deployment with autoscaling

A single Auto Scaling Group of self-contained instances behind an ALB, scaling
on inference queue depth, pre-warmed before the two daily traffic bands.

Read [PERFORMANCE.md](PERFORMANCE.md) first for where the capacity numbers come
from. The short version: **100 concurrent under 500 ms requires 200 req/s**, and
that is only practical on CPU with the `fast` tier.

---

## 1. Architecture

```
                    Internet
                       │
              ┌────────▼────────┐
              │       ALB       │  :443  health check → :8000/healthz
              └────────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
   │  EC2    │    │  EC2    │    │  EC2    │   Auto Scaling Group
   │ ┌─────┐ │    │ ┌─────┐ │    │ ┌─────┐ │   (c7i.2xlarge)
   │ │ api │ │    │ │ api │ │    │ │ api │ │
   │ └──┬──┘ │    │ └──┬──┘ │    │ └──┬──┘ │
   │ ┌──▼──┐ │    │ ┌──▼──┐ │    │ ┌──▼──┐ │
   │ │ ml  │ │    │ │ ml  │ │    │ │ ml  │ │
   │ └─────┘ │    │ └─────┘ │    │ └─────┘ │
   └─────────┘    └─────────┘    └─────────┘
      localhost       localhost      localhost
```

**Both containers run on every instance**, talking over localhost. This is the
arrangement that measured fastest, and it is also the easiest to scale:

- **Two processes, not one.** Everything in a single process contends for one
  GIL — measured 39.6 req/s versus 52.3 for the split. The localhost hop costs
  less than the contention it avoids.
- **No internal load balancer.** The API talks to `127.0.0.1:8001`, so there is
  no second ALB, no service discovery, and no cross-AZ traffic on the hot path.
- **One thing to scale.** Each instance is a complete, independent unit of
  capacity. Adding one adds a known number of req/s.
- **No partial-capacity states.** Separate API and ML groups can scale out of
  proportion and leave one tier starved; here they cannot drift apart.

Do not put the ML service behind its own ALB unless you genuinely need to scale
the tiers at different ratios. At this workload's shape you do not — the API is
I/O-bound and nearly free, and all the cost is inference.

---

## 2. Instance selection

| Instance | vCPU | RAM | Est. req/s (`fast`) | Notes |
|---|---|---|---|---|
| `c7i.xlarge` | 4 | 8 GB | ~25 | Finest scaling step; model memory duplicated per instance |
| **`c7i.2xlarge`** | **8** | **16 GB** | **~50** | **Recommended scaling unit** |
| `c7i.4xlarge` | 16 | 32 GB | ~100 | Fewer instances, coarser steps, bigger blast radius |
| `g5.xlarge` | 4 + A10G | 16 GB | 200+ | Only route to `accurate` tier at this deadline |

**Recommended: `c7i.2xlarge`.** Compute-optimized Sapphire Rapids — AVX-512 and
AMX are what ONNX Runtime's CPU kernels lean on for exactly this shape of work.
8 vCPU is the sweet spot between scaling granularity and the ~1–2 GB of model
weights each instance holds resident. 16 GB RAM leaves ample headroom.

**Graviton alternative:** `c7g.2xlarge` is ~15–20% cheaper and arm64 wheels for
onnxruntime exist. Benchmark it with `scripts/loadtest.py` before committing —
the x86 wheel path is the better-tested one.

**Avoid burstable (`t3`/`t4g`).** Sustained inference exhausts CPU credits and
the instance silently throttles, which looks exactly like a capacity problem and
is not.

### Fleet sizing for 200 req/s

| | Instances | vCPU | Note |
|---|---|---|---|
| Peak (100 concurrent @ <500 ms) | **5 × c7i.2xlarge** | 40 | 4 for 200 req/s + 1 for headroom/AZ loss |
| Busy-hour average (~24 req/s) | 2 | 16 | Matches today's real peak minute |
| Off-peak (~1–4 req/s) | 1 | 8 | Floor for availability |

Spread across **3 AZs** so losing one costs a third of capacity, not half.

> Sizing assumes ~50 req/s per `c7i.2xlarge` on the `fast` tier, extrapolated
> from an 8-core benchmark on different silicon. **Measure one instance before
> buying five** — see §8.

---

## 3. Bake an AMI

Cold start decides how fast autoscaling can respond. With models already on
disk, the service is ready in ~5 seconds; pulling `buffalo_l` (~300 MB) on first
boot instead makes that 60–120 s — far too slow to answer a burst.

Bake everything into the AMI: Docker, images, and model weights.

```bash
# On a build instance from Ubuntu 22.04 LTS
sudo apt-get update && sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl enable docker

# Application + images
sudo git clone <your-repo-url> /opt/zepiris
cd /opt/zepiris
sudo docker compose build

# Pre-download model packs into the volume the ML container mounts.
# Both packs: buffalo_l supplies ResNet50 recognition, buffalo_s the small detector.
sudo docker run --rm -v insightface_cache:/root/.insightface zepiris-ml-inference \
  python -c "
from insightface.utils import storage
for pack in ('buffalo_l', 'buffalo_s'):
    storage.ensure_available('models', pack, root='/root/.insightface')
    print('cached', pack)
"

# Verify it boots and is ready before capturing the image
sudo docker compose up -d
timeout 180 bash -c 'until curl -fsS http://localhost:8001/readyz; do sleep 2; done'
curl -fsS http://localhost:8000/healthz
sudo docker compose down
```

Then capture the AMI:

```bash
aws ec2 create-image --instance-id i-BUILDER --name "zepiris-$(date +%Y%m%d-%H%M)" \
  --description "ZepIris API + ML, models pre-cached" --no-reboot
```

Re-bake on every release. An AMI whose model cache is stale is an instance that
downloads 300 MB mid-burst.

---

## 4. Launch template

`user-data` starts the stack and nothing else — all the slow work is in the AMI.

```bash
#!/bin/bash
set -euxo pipefail
cd /opt/zepiris

cat > .env <<'EOF'
# --- ML service ---
ML_SERVICE_FACE_MATCH_ONLY=true
# "fast" is the only CPU tier that reaches 200 req/s. Its scores are NOT
# comparable to the other tiers — recalibrate the threshold before switching.
ML_SERVICE_FACE_TIER=fast
ML_SERVICE_FACE_INTRA_OP_THREADS=1
ML_SERVICE_FACE_MAX_INPUT_SIDE=1600
# 0 = CPU count - 1. Leaves a core for the event loop so the instance can still
# accept and shed traffic when saturated.
ML_SERVICE_MAX_CONCURRENT_INFERENCES=0
# Shed rather than queue past the caller's own timeout.
ML_SERVICE_INFERENCE_QUEUE_TIMEOUT_SECONDS=5
ML_SERVICE_WARMUP_ON_STARTUP=true

# --- API ---
ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://ml-inference:8001
ZEPIRIS_VERIFY_THRESHOLD=0.5
ZEPIRIS_ML_MAX_CONNECTIONS=200
ZEPIRIS_S3_MAX_CONNECTIONS=200
ZEPIRIS_API_WORKERS=2
# Threshold learning appends to a local file. On an autoscaled fleet that file
# dies with the instance, so keep it off unless it is on shared storage.
ZEPIRIS_LEARNING_ENABLED=false
EOF

docker compose up -d

# Publish queue depth to CloudWatch — the signal the ASG scales on (§6).
cat > /usr/local/bin/publish-metrics.sh <<'EOS'
#!/bin/bash
IID=$(curl -sf -H "X-aws-ec2-metadata-token: $(curl -sf -X PUT \
  http://169.254.169.254/latest/api/token \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" \
  http://169.254.169.254/latest/meta-data/instance-id)
while true; do
  BODY=$(curl -sf --max-time 3 http://localhost:8001/metrics) || { sleep 10; continue; }
  QD=$(echo "$BODY" | python3 -c 'import sys,json; print(json.load(sys.stdin)["inference"]["queue_depth"])' 2>/dev/null) || { sleep 10; continue; }
  aws cloudwatch put-metric-data --namespace ZepIris \
    --metric-name InferenceQueueDepth --value "$QD" --unit Count \
    --dimensions AutoScalingGroupName=zepiris-asg --region ap-south-1 || true
  sleep 20
done
EOS
chmod +x /usr/local/bin/publish-metrics.sh
nohup /usr/local/bin/publish-metrics.sh >/var/log/zepiris-metrics.log 2>&1 &
```

Create it:

```bash
aws ec2 create-launch-template --launch-template-name zepiris-lt \
  --version-description v1 \
  --launch-template-data '{
    "ImageId": "ami-YOURBAKEDAMI",
    "InstanceType": "c7i.2xlarge",
    "IamInstanceProfile": {"Name": "zepiris-instance-profile"},
    "SecurityGroupIds": ["sg-INSTANCE"],
    "MetadataOptions": {"HttpTokens": "required"},
    "BlockDeviceMappings": [{
      "DeviceName": "/dev/sda1",
      "Ebs": {"VolumeSize": 40, "VolumeType": "gp3", "DeleteOnTermination": true}
    }],
    "UserData": "'"$(base64 -w0 user-data.sh)"'"
  }'
```

The instance role needs `cloudwatch:PutMetricData`, plus S3 read for your
reference-image bucket if you use IAM rather than presigned URLs.

**Security group:** ALB reaches `:8000` only. Port `8001` must not be reachable
from outside the instance — the API talks to it over localhost.

---

## 5. ALB and target group

Health-check `/healthz` on the API. It only reports OK once the API is up, and
the API's own dependency, the ML container, gates itself on `/readyz` — which
stays 503 until the models are loaded **and** warmed.

```bash
aws elbv2 create-target-group --name zepiris-tg \
  --protocol HTTP --port 8000 --vpc-id vpc-XXXX --target-type instance \
  --health-check-path /healthz \
  --health-check-interval-seconds 10 \
  --health-check-timeout-seconds 5 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 3 \
  --matcher HttpCode=200
```

Two settings matter more than they look:

```bash
# Finish in-flight verifications before an instance leaves. Scale-in during a
# burst would otherwise kill requests that were already being served.
aws elbv2 modify-target-group-attributes --target-group-arn <TG_ARN> \
  --attributes Key=deregistration_delay.timeout_seconds,Value=30

# Least-outstanding-requests, not round-robin. Inference times vary, and
# round-robin will hand a request to an instance that is already saturated
# while another sits idle.
aws elbv2 modify-target-group-attributes --target-group-arn <TG_ARN> \
  --attributes Key=load_balancing.algorithm.type,Value=least_outstanding_requests
```

Set the ALB idle timeout above your client timeout (60 s is fine).

---

## 6. Auto Scaling Group

```bash
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name zepiris-asg \
  --launch-template LaunchTemplateName=zepiris-lt,Version='$Latest' \
  --min-size 1 --max-size 8 --desired-capacity 2 \
  --vpc-zone-identifier "subnet-AZ1,subnet-AZ2,subnet-AZ3" \
  --target-group-arns <TG_ARN> \
  --health-check-type ELB \
  --health-check-grace-period 120 \
  --default-instance-warmup 90
```

`--default-instance-warmup 90` is what stops the classic autoscaling failure:
without it, a booting instance reports near-zero load, drags the group average
down, and the ASG concludes it can scale *in* — mid-burst.

### Warm pool: the difference between 3 minutes and 30 seconds

Even with a baked AMI, a cold EC2 launch is ~2–3 minutes before traffic flows.
Their traffic peaks inside a *single minute*. A warm pool keeps pre-initialized
instances **stopped** — you pay only for EBS — and starting one takes seconds.

```bash
aws autoscaling put-warm-pool --auto-scaling-group-name zepiris-asg \
  --min-size 3 --pool-state Stopped \
  --instance-reuse-policy '{"ReuseOnScaleIn": true}'
```

Three stopped `c7i.2xlarge` cost a few dollars a month in EBS and turn a
3-minute scale-out into a ~30-second one.

### Scaling policies

**Primary — target tracking on queue depth.** Scale on the signal that
distinguishes "busy" from "overloaded". CPU cannot: it pins near 100% in both
cases. Queue depth is zero whenever the instance is keeping up, and positive the
moment it is not.

```bash
aws autoscaling put-scaling-policy --auto-scaling-group-name zepiris-asg \
  --policy-name zepiris-queue-depth --policy-type TargetTrackingScaling \
  --estimated-instance-warmup 90 \
  --target-tracking-configuration '{
    "TargetValue": 1.0,
    "CustomizedMetricSpecification": {
      "MetricName": "InferenceQueueDepth",
      "Namespace": "ZepIris",
      "Dimensions": [{"Name": "AutoScalingGroupName", "Value": "zepiris-asg"}],
      "Statistic": "Average"
    }
  }'
```

A target of 1.0 means "on average, at most one request waiting per instance" —
tight enough to protect the deadline, loose enough not to flap on jitter.

**Secondary — request count per target.** A safety net that reacts even if the
metric publisher dies:

```bash
aws autoscaling put-scaling-policy --auto-scaling-group-name zepiris-asg \
  --policy-name zepiris-rpt --policy-type TargetTrackingScaling \
  --estimated-instance-warmup 90 \
  --target-tracking-configuration '{
    "TargetValue": 35.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ALBRequestCountPerTarget",
      "ResourceLabel": "app/zepiris-alb/ID/targetgroup/zepiris-tg/ID"
    }
  }'
```

35 req/s per target against ~50 capacity is ~70% utilization — headroom to
absorb a burst while new instances boot. Multiple target-tracking policies
coexist safely: the ASG takes the largest capacity any of them asks for, and
scales in only when all agree.

**Scheduled pre-warm.** Reactive scaling cannot beat a burst that arrives inside
one minute. Traffic has two known bands, so scale ahead of them:

```bash
# 05:50 IST (00:20 UTC) — before the 06:00 shift-start spike
aws autoscaling put-scheduled-update-group-action --auto-scaling-group-name zepiris-asg \
  --scheduled-action-name prewarm-morning --recurrence "20 0 * * *" \
  --min-size 5 --max-size 8

# 08:30 IST — morning band over
aws autoscaling put-scheduled-update-group-action --auto-scaling-group-name zepiris-asg \
  --scheduled-action-name scalein-morning --recurrence "0 3 * * *" \
  --min-size 2 --max-size 8

# 13:50 IST — before the afternoon band
aws autoscaling put-scheduled-update-group-action --auto-scaling-group-name zepiris-asg \
  --scheduled-action-name prewarm-evening --recurrence "20 8 * * *" \
  --min-size 5 --max-size 8

# 18:45 IST — evening band over
aws autoscaling put-scheduled-update-group-action --auto-scaling-group-name zepiris-asg \
  --scheduled-action-name scalein-evening --recurrence "15 13 * * *" \
  --min-size 1 --max-size 8
```

Scheduled actions move `min-size`; target tracking stays free to go higher. The
floor is guaranteed capacity, not a ceiling.

**Scale-in protection.** Terminate the newest instance, not a warm one serving
traffic:

```bash
aws autoscaling put-scaling-policy --auto-scaling-group-name zepiris-asg \
  --policy-name zepiris-termination \
  --policy-type SimpleScaling --scaling-adjustment -1 \
  --adjustment-type ChangeInCapacity --cooldown 300
```

A 300 s scale-in cooldown against a 90 s warmup means the group expands quickly
and contracts slowly — the right asymmetry when the cost of being one instance
short is a wave of failed verifications.

---

## 7. Cost

`ap-south-1` on-demand, `c7i.2xlarge` ≈ $0.42/hr.

| Component | Hours/day | Cost/month |
|---|---|---|
| Baseline 1–2 instances (24/7) | 24 | ~$300–600 |
| Peak bands, +3 instances (~7.5 h/day) | 7.5 | ~$280 |
| Warm pool (3 stopped, EBS only) | — | ~$10 |
| ALB | 24 | ~$25 |
| **Total (on-demand)** | | **~$615–915** |

**Use Spot for the peak-band instances.** Bursts are interruption-tolerant when
the baseline is on-demand, and Spot runs ~70% cheaper — bringing the total to
roughly **$400–500/month**. Configure a mixed-instances policy with on-demand
covering the base capacity:

```bash
aws autoscaling update-auto-scaling-group --auto-scaling-group-name zepiris-asg \
  --mixed-instances-policy '{
    "LaunchTemplate": {
      "LaunchTemplateSpecification": {"LaunchTemplateName": "zepiris-lt", "Version": "$Latest"},
      "Overrides": [
        {"InstanceType": "c7i.2xlarge"},
        {"InstanceType": "c6i.2xlarge"},
        {"InstanceType": "c7a.2xlarge"}
      ]
    },
    "InstancesDistribution": {
      "OnDemandBaseCapacity": 2,
      "OnDemandPercentageAboveBaseCapacity": 0,
      "SpotAllocationStrategy": "capacity-optimized"
    }
  }'
```

Listing several instance types widens the Spot pool and cuts interruption rates
materially.

---

## 8. Validate before you trust it

**Measure one instance first.** Every number above extrapolates from a benchmark
on different silicon:

```bash
# On one instance, against localhost — no ALB in the path
python scripts/loadtest.py --target api --concurrency 16 --requests 300
```

Read the throughput. Then `instances = ceil(200 / that)`. If a `c7i.2xlarge`
does 60 req/s, four instances suffice; if it does 35, you need six.

**Then test the fleet through the ALB:**

```bash
python scripts/loadtest.py --base-url https://your-alb-dns \
  --concurrency 100 --requests 2000 --deadline-ms 500
```

**Then test that scaling actually works** — the part that is usually broken and
nobody notices until it matters:

```bash
# Scale in to the floor, then apply peak load and watch recovery
aws autoscaling set-desired-capacity --auto-scaling-group-name zepiris-asg --desired-capacity 1
python scripts/loadtest.py --base-url https://your-alb-dns --concurrency 100 --requests 5000
aws autoscaling describe-scaling-activities --auto-scaling-group-name zepiris-asg --max-items 10
```

What to confirm:
- New instances appear within ~30 s (warm pool) or ~2–3 min (cold).
- Overload returns **503s, not timeouts** — shedding is working.
- p95 recovers under 500 ms once the group settles.
- The group scales back *down* afterwards, and does not oscillate.

---

## 9. Alarms worth having

| Alarm | Condition | Why |
|---|---|---|
| Sustained queue depth | `InferenceQueueDepth > 3` for 5 min | Scaling is not keeping up |
| Shedding | ALB `HTTPCode_Target_5XX_Count > 0` for 5 min | Genuinely over capacity |
| Latency | ALB `TargetResponseTime` p95 > 0.5 s for 5 min | The actual SLO |
| Unhealthy hosts | `UnHealthyHostCount > 0` for 5 min | Warm-up failing, or a bad AMI |
| At ceiling | `GroupInServiceInstances >= max-size` | Raise `max-size` before it bites |
| Spot reclaims | Spot interruption rate rising | Rebalance the instance-type mix |

The one to page on is **latency**, because it is the promise being made. The
others explain it.

---

## 10. Deploying a new version

Rolling instance refresh, with a minimum healthy percentage so capacity never
dips below the current load:

```bash
aws autoscaling start-instance-refresh --auto-scaling-group-name zepiris-asg \
  --preferences '{
    "MinHealthyPercentage": 75,
    "InstanceWarmup": 90,
    "ScaleInProtectedInstances": "Ignore",
    "SkipMatching": false
  }'
```

Never refresh during a peak band — `MinHealthyPercentage: 75` still removes a
quarter of the fleet, and a shift-start burst has no room for that. Deploy
between bands (09:00–13:00 or after 19:00 IST).
