# EC2 deployment with autoscaling

A single Auto Scaling Group of self-contained instances behind an ALB, scaling
on inference queue depth, pre-warmed before the two daily traffic bands.

**Launch target: 100 req/s on the `balanced` tier — 6 instances at peak, 1 off
peak, thresholds unchanged.**

Read [PERFORMANCE.md](PERFORMANCE.md) for where the capacity numbers come from.
The rule behind all of them: sustainable concurrency = throughput x deadline, so
100 req/s against a 500 ms deadline means roughly 50 requests in flight at once.

> **Considering GPU instead?** At 100 req/s two `g4dn.xlarge` (T4) cost roughly
> half this CPU fleet and run the `accurate` model — see
> [CPU_VS_GPU.md](CPU_VS_GPU.md). Everything in this document (ASG, ALB, warm
> pool, scaling signal) applies unchanged; only the instance type, the image, and
> two settings differ. This plan is the option that ships with no unvalidated
> assumptions in it.

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

| Instance | vCPU | RAM | Est. req/s (`balanced`) | Notes |
|---|---|---|---|---|
| `c7i.xlarge` | 4 | 8 GB | ~9 | Finest scaling step; model memory duplicated per instance |
| **`c7i.2xlarge`** | **8** | **16 GB** | **~17.5** | **Recommended scaling unit** |
| `c7i.4xlarge` | 16 | 32 GB | ~35 | Fewer instances, coarser steps, bigger blast radius |
| `g5.2xlarge` | 8 + A10G | 32 GB | 100+ | Overkill at this target; revisit past ~500 req/s |

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

### Fleet sizing for 100 req/s — the launch target

At 100 req/s the interesting question is not how few instances you can run. It is
whether you can afford to keep the **accurate recognition weights**, and the
answer is yes.

| Tier | req/s per instance | Instances for 100 req/s | +1 for AZ loss | Cost/mo at peak | Threshold work |
|---|---|---|---|---|---|
| `accurate` | ~9.8 | 11 | 12 | ~$3,700 | none |
| **`balanced`** ← recommended | **~17.5** | **6** | **7** | **~$2,150** | **none** |
| `fast` | ~52 | 2 | 3 | ~$920 | recalibration required |

**Run `balanced`.** It keeps buffalo_l's ResNet50 recognition — the model that
produces the embedding and decides every match — so existing thresholds carry
over untouched and no recalibration campaign stands between you and launch. Seven
instances is an ordinary fleet.

`fast` saves ~$1,200/month at peak, but it moves the embedding space and every
threshold has to be re-fit against labelled pairs first. On a biometric decision
system that is a poor trade at this scale. It is the right answer at 1000 req/s,
where the CPU alternative stops being affordable — not here.

### What you will actually pay

Peak sizing is not the bill. Real traffic peaks at ~24 req/s in the busiest
minute and sits near 1 req/s off-peak, so the group spends most of the day at its
floor:

| Period | Instances | Hours/day |
|---|---|---|
| Peak bands (pre-warmed) | 6–7 | ~7.5 |
| Busy-hour average | 2–3 | ~6 |
| Off-peak | 1 | ~10.5 |

Blended, that is roughly **$740–1,040/month on-demand**, or **$450–600 with
Spot** for the peak-band instances (§7 breaks this down). The 100 req/s figure is
headroom — about 4x the busiest minute observed — so the fleet is sized for a
burst it will rarely see.

### Confidence in these numbers

Derived from a clean full-path measurement of the `fast` tier (52 req/s on 8
cores) plus per-tier model cost measured single-threaded, on an **8-core Apple
Silicon machine with mixed performance and efficiency cores**. Uniform x86 cores
should do better, so treat this as conservative — you are more likely to need
fewer instances than more.

Measure before you commit:

```bash
python scripts/loadtest.py --target api --concurrency 16 --requests 300
```

Then `instances = ceil(100 / measured)`. If a `c7i.2xlarge` does 25 req/s on
`balanced`, four instances carry the target and the recommendation above is one
instance too cautious.

### Free capacity before adding instances

Capturing selfies at 640x854 instead of 1800x2400 is ~2.4x throughput at no
accuracy cost — which would take the `balanced` fleet from 6 instances to 3. If
the mobile client can be changed, do that first. See
[PERFORMANCE.md](PERFORMANCE.md).

Spread the fleet across **3 AZs** so losing one costs a third of capacity, not
half.

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
# "balanced" keeps buffalo_l's ResNet50 recognition, so existing thresholds carry
# over unchanged — the right choice at 100 req/s. "fast" is ~3x cheaper again but
# moves the embedding space and REQUIRES recalibration; save it for the scale
# where the CPU alternative stops being affordable.
ML_SERVICE_FACE_TIER=balanced
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
  --min-size 1 --max-size 10 --desired-capacity 2 \
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
  --min-size 5 --pool-state Stopped \
  --instance-reuse-policy '{"ReuseOnScaleIn": true}'
```

Five stopped `c7i.2xlarge` cost a few dollars a month in EBS and turn a
3-minute scale-out into a ~30-second one. Size the pool to cover the jump from
the off-peak floor to a full peak band, which here is 1 -> 6.

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
    "TargetValue": 12.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ALBRequestCountPerTarget",
      "ResourceLabel": "app/zepiris-alb/ID/targetgroup/zepiris-tg/ID"
    }
  }'
```

12 req/s per target against ~17.5 capacity on the `balanced` tier is ~70%
utilization — headroom to absorb a burst while new instances boot. Raise it to
~35 if you later move to the `fast` tier. Multiple target-tracking policies
coexist safely: the ASG takes the largest capacity any of them asks for, and
scales in only when all agree.

**Scheduled pre-warm.** Reactive scaling cannot beat a burst that arrives inside
one minute. Traffic has two known bands, so scale ahead of them:

```bash
# 05:50 IST (00:20 UTC) — before the 06:00 shift-start spike
aws autoscaling put-scheduled-update-group-action --auto-scaling-group-name zepiris-asg \
  --scheduled-action-name prewarm-morning --recurrence "20 0 * * *" \
  --min-size 6 --max-size 10

# 08:30 IST — morning band over
aws autoscaling put-scheduled-update-group-action --auto-scaling-group-name zepiris-asg \
  --scheduled-action-name scalein-morning --recurrence "0 3 * * *" \
  --min-size 2 --max-size 10

# 13:50 IST — before the afternoon band
aws autoscaling put-scheduled-update-group-action --auto-scaling-group-name zepiris-asg \
  --scheduled-action-name prewarm-evening --recurrence "20 8 * * *" \
  --min-size 6 --max-size 10

# 18:45 IST — evening band over
aws autoscaling put-scheduled-update-group-action --auto-scaling-group-name zepiris-asg \
  --scheduled-action-name scalein-evening --recurrence "15 13 * * *" \
  --min-size 1 --max-size 10
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
| Peak bands, +4–5 instances (~7.5 h/day) | 7.5 | ~$400 |
| Warm pool (5 stopped, EBS only) | — | ~$15 |
| ALB | 24 | ~$25 |
| **Total (on-demand)** | | **~$740–1,040** |

**Use Spot for the peak-band instances.** Bursts are interruption-tolerant when
the baseline is on-demand, and Spot runs ~70% cheaper — bringing the total to
roughly **$450–600/month**. Configure a mixed-instances policy with on-demand
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

Read the throughput. Then `instances = ceil(100 / that)`. If a `c7i.2xlarge`
does 25 req/s on `balanced`, four instances carry the target; if it does 15,
you need seven.

**Then test the fleet through the ALB:**

```bash
# 100 req/s against a 500 ms deadline is ~50 requests in flight
python scripts/loadtest.py --base-url https://your-alb-dns \
  --concurrency 50 --requests 2000 --deadline-ms 500
```

**Then test that scaling actually works** — the part that is usually broken and
nobody notices until it matters:

```bash
# Scale in to the floor, then apply peak load and watch recovery
aws autoscaling set-desired-capacity --auto-scaling-group-name zepiris-asg --desired-capacity 1
python scripts/loadtest.py --base-url https://your-alb-dns --concurrency 50 --requests 5000
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
