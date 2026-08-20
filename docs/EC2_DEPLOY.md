# Deploying ZepIris on EC2 (Docker Compose)

Two stateless containers, nothing persisted except the InsightFace model cache:

```
api (8000)  ──►  ml-inference (8001)
   └─ fetches reference images from S3 URLs at request time
```

---

## 1. Provision the EC2 instance

| Setting        | Recommendation                                                        |
|----------------|-----------------------------------------------------------------------|
| AMI            | Ubuntu 22.04 LTS (or Amazon Linux 2023)                               |
| Instance type  | `t3.large` (2 vCPU / 8 GB) minimum; `t3.xlarge` (16 GB) comfortable   |
| Storage        | 30 GB gp3 EBS (PyTorch image + models + buffalo_l cache need room)    |
| Network        | Public IP (or behind ALB); **outbound internet required**             |

**Why outbound internet matters:** on first boot the ML service downloads the
InsightFace `buffalo_l` models (~300 MB), and the API fetches reference images
from your S3 URLs at request time.

### Security Group (inbound)

| Port | Source                | Purpose                                   |
|------|-----------------------|-------------------------------------------|
| 22   | your IP               | SSH                                       |
| 8000 | your app / ALB / your IP | ZepIris API                            |
| 8001 | **none** (leave closed) | ML service — internal to the box only   |

> The API talks to the ML service over the Docker network, so 8001 never needs
> to be exposed publicly.

---

## 2. Install Docker + Compose v2

```bash
# Ubuntu
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker   # or log out/in so the group takes effect
```

Verify: `docker compose version`

---

## 3. Get the code onto EC2

The repo ships ~105 MB of model weights in `models/`. Two options:

**A. git (if models are committed via Git LFS or normal git):**
```bash
git clone <your-repo-url> zepiris
cd zepiris
```

**B. scp from your machine (no remote git needed):**
```bash
# from your laptop, in the parent dir of zepiris/
rsync -avz --exclude '.venv' --exclude '.git' --exclude 'logs' \
  zepiris/ ubuntu@<EC2_PUBLIC_IP>:~/zepiris/
```

Either way you must end up with `models/*.pth` and `models/*.onnx` present on EC2 —
they are `COPY`'d into the ML image at build time.

---

## 4. Build and run

```bash
cd ~/zepiris
docker compose up -d --build      # first build pulls PyTorch — takes several minutes
docker compose ps                 # both should become healthy
docker compose logs -f ml-inference   # watch buffalo_l download on first boot
```

Health checks:
```bash
curl http://localhost:8000/healthz     # {"status":"ok"}
curl http://localhost:8001/healthz     # {"status":"ok"}  (localhost only)
```

Smoke-test verify (JSON body; use presigned/public S3 URLs):
```bash
curl -X POST http://localhost:8000/v1/faces/facematch/verify \
  -H "Content-Type: application/json" \
  -d '{
    "face_check_s3": "https://your-bucket.s3.amazonaws.com/incoming.jpg?X-Amz-Signature=...",
    "source_selfie_s3": "https://your-bucket.s3.amazonaws.com/enrolled.jpg?X-Amz-Signature=..."
  }'
```

The match is **probe vs enrolled selfie**: `source_selfie_*` is the enrolled
selfie (source of truth, only embedded); the probe is the incoming
`face_check_*` (live face, facematch) or `doc_check_*` (ID document, docmatch).
Each side accepts **base64 or an S3 URL**. See `docs/API_REFERENCE.md`.

`restart: always` is already set, so both containers come back after a reboot
(Docker's service is enabled by default on Ubuntu).

---

## 5. HTTPS in front of the API (optional)

The service speaks plain HTTP on 8000. That is fine when the caller is your own
backend inside the VPC. Terminate TLS when the caller is outside it:

- **ALB / API Gateway** — the usual answer if you already run one; point it at
  8000 and keep the security group closed to everything else.
- **Caddy on the same box** — quickest for a single instance (auto Let's Encrypt):

  ```bash
  # /etc/caddy/Caddyfile
  api.yourdomain.com {
      reverse_proxy 127.0.0.1:8000
  }
  ```
  Point a DNS A record at the EC2 IP, open 80/443 in the security group, and Caddy
  provisions a cert automatically.

---

## 6. Tuning the match threshold

Default cosine match threshold is `0.5` (`ZEPIRIS_VERIFY_THRESHOLD` in
`docker-compose.yml`). Raise it for stricter matching, lower for more lenient.
Callers can also override per-request with the `threshold` form field.

---

## Updating a deployment

```bash
cd ~/zepiris
git pull            # or re-rsync
docker compose up -d --build
```

The `insightface_cache` volume persists across rebuilds, so buffalo_l is not
re-downloaded.
