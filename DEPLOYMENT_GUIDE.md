# WASM-Torch Production Deployment Guide

🚀 **Complete deployment guide for WASM-Torch in production environments**

## 📋 Prerequisites

### System Requirements
- **Python**: 3.10+ 
- **Node.js**: 16+ (for browser runtime)
- **Docker**: 20+ (for containerized deployment)
- **Kubernetes**: 1.25+ (for orchestrated deployment)
- **Memory**: 4GB+ RAM recommended
- **Storage**: 10GB+ available space

### Build Dependencies
```bash
# Core dependencies
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    python3-dev \
    nodejs \
    npm

# Emscripten toolchain (for WASM compilation)
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

## 🏗️ Installation Methods

### Method 1: PyPI Installation (Recommended)
```bash
# Install from PyPI
pip install wasm-shim-torch

# Verify installation
wasm-torch --version
wasm-torch check-system
```

### Method 2: Source Installation
```bash
# Clone repository
git clone https://github.com/yourusername/wasm-shim-for-torch.git
cd wasm-shim-for-torch

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/ -v
```

### Method 3: Docker Deployment
```bash
# Build Docker image
docker build -t wasm-torch:latest .

# Run container
docker run -p 8000:8000 \
  -e WASM_TORCH_LOG_LEVEL=info \
  -e WASM_TORCH_MAX_MEMORY_MB=2048 \
  wasm-torch:latest
```

## 🚢 Production Deployment Options

### Option A: Kubernetes Deployment

#### 1. Deploy Base Infrastructure
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s-deployment.yml

# Verify deployment
kubectl get pods -l app=wasm-torch
kubectl logs -l app=wasm-torch
```

#### 2. Configure Load Balancing
```yaml
# load-balancer.yml
apiVersion: v1
kind: Service
metadata:
  name: wasm-torch-lb
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: wasm-torch
```

#### 3. Setup Auto-scaling
```yaml
# hpa.yml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: wasm-torch-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: wasm-torch
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Option B: Cloud Service Deployment

#### AWS Lambda (Serverless)
```bash
# Package for Lambda
pip install --target ./package -r requirements.txt
cd package && zip -r ../deployment.zip .
cd .. && zip -g deployment.zip lambda_function.py

# Deploy using AWS CLI
aws lambda create-function \
  --function-name wasm-torch-inference \
  --runtime python3.10 \
  --role arn:aws:iam::ACCOUNT:role/lambda-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://deployment.zip \
  --memory-size 1024 \
  --timeout 30
```

#### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/wasm-torch
gcloud run deploy wasm-torch \
  --image gcr.io/PROJECT-ID/wasm-torch \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 100
```

#### Azure Container Instances
```bash
# Deploy to ACI
az container create \
  --resource-group myResourceGroup \
  --name wasm-torch-instance \
  --image yourdockerhub/wasm-torch:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables \
    WASM_TORCH_LOG_LEVEL=info \
    WASM_TORCH_MAX_WORKERS=4
```

## ⚙️ Configuration Management

### Environment Variables
```bash
# Core Configuration
export WASM_TORCH_LOG_LEVEL=info              # debug, info, warning, error
export WASM_TORCH_MAX_MEMORY_MB=2048          # Maximum memory usage
export WASM_TORCH_MAX_WORKERS=4               # Worker pool size
export WASM_TORCH_CACHE_SIZE_MB=512           # Cache size
export WASM_TORCH_ENABLE_SIMD=true            # Enable SIMD optimizations
export WASM_TORCH_ENABLE_WEBGPU=true          # Enable WebGPU acceleration

# Security Configuration
export WASM_TORCH_ENABLE_AUTH=true            # Enable authentication
export WASM_TORCH_JWT_SECRET=your-secret-key  # JWT signing secret
export WASM_TORCH_RATE_LIMIT=100              # Requests per minute
export WASM_TORCH_CORS_ORIGINS="*"            # CORS allowed origins

# Performance Configuration
export WASM_TORCH_PRELOAD_MODELS=true         # Enable model preloading
export WASM_TORCH_BATCH_SIZE=16               # Default batch size
export WASM_TORCH_TIMEOUT_SECONDS=30          # Request timeout
export WASM_TORCH_HEALTH_CHECK_INTERVAL=30    # Health check interval

# Monitoring Configuration
export WASM_TORCH_ENABLE_METRICS=true         # Enable Prometheus metrics
export WASM_TORCH_METRICS_PORT=9090           # Metrics endpoint port
export WASM_TORCH_ENABLE_TRACING=true         # Enable request tracing
export WASM_TORCH_JAEGER_ENDPOINT=localhost:14268  # Tracing endpoint
```

### Configuration File
```json
// config/production.json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4
  },
  "runtime": {
    "max_memory_mb": 2048,
    "enable_simd": true,
    "enable_webgpu": true,
    "cache_size_mb": 512
  },
  "scaling": {
    "min_workers": 2,
    "max_workers": 20,
    "target_queue_size": 5,
    "auto_optimization": true
  },
  "reliability": {
    "max_retries": 3,
    "circuit_breaker_threshold": 5,
    "health_check_interval": 30
  },
  "security": {
    "enable_auth": true,
    "rate_limit_rpm": 100,
    "cors_origins": ["https://yourdomain.com"],
    "validate_inputs": true
  },
  "monitoring": {
    "enable_metrics": true,
    "enable_tracing": true,
    "log_level": "info"
  }
}
```

## 📊 Monitoring & Observability

### Metrics Collection
```bash
# Prometheus metrics available at /metrics
curl http://localhost:8000/metrics

# Key metrics to monitor:
# - wasm_torch_inferences_total
# - wasm_torch_inference_duration_seconds
# - wasm_torch_memory_usage_bytes
# - wasm_torch_error_rate
# - wasm_torch_cache_hit_rate
```

### Health Checks
```bash
# Health check endpoint
curl http://localhost:8000/health

# Readiness check
curl http://localhost:8000/ready

# Detailed status
curl http://localhost:8000/status
```

### Logging Configuration
```yaml
# logging.yml
version: 1
formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
  json:
    format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: json
    stream: ext://sys.stdout
    
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: standard
    filename: /var/log/wasm-torch/app.log
    maxBytes: 10485760
    backupCount: 5

loggers:
  wasm_torch:
    level: INFO
    handlers: [console, file]
    propagate: false

root:
  level: WARNING
  handlers: [console]
```

## 🔐 Security Hardening

### TLS/SSL Configuration
```nginx
# nginx.conf for HTTPS termination
server {
    listen 443 ssl http2;
    server_name api.wasm-torch.com;
    
    ssl_certificate /etc/ssl/certs/wasm-torch.crt;
    ssl_certificate_key /etc/ssl/private/wasm-torch.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    
    location / {
        proxy_pass http://wasm-torch-backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Security headers
        add_header X-Content-Type-Options nosniff;
        add_header X-Frame-Options DENY;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    }
}
```

### Authentication & Authorization
```python
# Example API key authentication
from wasm_torch.security import APIKeyAuth

auth = APIKeyAuth(
    api_keys={
        "prod-key-1": {"scope": "inference", "rate_limit": 1000},
        "admin-key": {"scope": "admin", "rate_limit": 10000}
    }
)

app = create_app(auth=auth)
```

### Input Validation
```python
# Request validation schema
INFERENCE_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "model_id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$"},
        "input_data": {
            "type": "array",
            "maxItems": 10000,
            "items": {"type": "number"}
        },
        "options": {
            "type": "object",
            "properties": {
                "timeout": {"type": "number", "minimum": 1, "maximum": 300},
                "batch_size": {"type": "number", "minimum": 1, "maximum": 64}
            }
        }
    },
    "required": ["model_id", "input_data"]
}
```

## 🔧 Performance Optimization

### Memory Management
```bash
# Optimize for memory usage
export MALLOC_MMAP_THRESHOLD_=65536
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_TOP_PAD_=131072
export MALLOC_MMAP_MAX_=65536
```

### CPU Optimization
```bash
# Set CPU affinity for better performance
taskset -c 0-3 python app.py

# Use performance CPU governor
echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### I/O Optimization
```bash
# Optimize for I/O performance
echo mq-deadline > /sys/block/*/queue/scheduler
echo 2 > /sys/block/*/queue/rq_affinity
```

## 📈 Scaling Strategies

### Horizontal Scaling
```yaml
# Kubernetes HPA with custom metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: wasm-torch-custom-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: wasm-torch
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Pods
    pods:
      metric:
        name: inference_queue_length
      target:
        type: AverageValue
        averageValue: "5"
  - type: Object
    object:
      metric:
        name: requests_per_second
      target:
        type: Value
        value: "100"
```

### Vertical Scaling
```bash
# VPA for automatic resource adjustment
kubectl apply -f - <<EOF
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: wasm-torch-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: wasm-torch
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: wasm-torch
      maxAllowed:
        cpu: 4
        memory: 8Gi
      minAllowed:
        cpu: 100m
        memory: 256Mi
EOF
```

## 🧪 Testing in Production

### Load Testing
```bash
# Use Apache Bench for load testing
ab -n 1000 -c 10 \
  -H "Content-Type: application/json" \
  -p test_payload.json \
  http://localhost:8000/api/v1/inference

# Use wrk for advanced load testing
wrk -t12 -c400 -d30s \
  --script=load_test.lua \
  http://localhost:8000/api/v1/inference
```

### Canary Deployment
```yaml
# Istio canary deployment
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: wasm-torch-canary
spec:
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: wasm-torch
        subset: v2
  - route:
    - destination:
        host: wasm-torch
        subset: v1
      weight: 90
    - destination:
        host: wasm-torch
        subset: v2
      weight: 10
```

### A/B Testing
```python
# Feature flag based A/B testing
from wasm_torch.experiments import ABTest

ab_test = ABTest("simd_optimization", {
    "control": {"enable_simd": False},
    "treatment": {"enable_simd": True}
}, traffic_split=0.5)

@app.route("/inference")
def inference():
    config = ab_test.get_config(request.user_id)
    return run_inference_with_config(config)
```

## 🚨 Incident Response

### Monitoring Alerts
```yaml
# Prometheus alert rules
groups:
- name: wasm-torch-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(wasm_torch_errors_total[5m]) > 0.1
    for: 2m
    annotations:
      summary: "High error rate detected"
      
  - alert: HighMemoryUsage
    expr: wasm_torch_memory_usage_bytes > 1.5e+9
    for: 5m
    annotations:
      summary: "Memory usage above 1.5GB"
      
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(wasm_torch_inference_duration_seconds_bucket[5m])) > 1.0
    for: 3m
    annotations:
      summary: "95th percentile latency above 1s"
```

### Recovery Procedures
```bash
# Automated recovery script
#!/bin/bash
# recovery.sh

echo "Starting WASM-Torch recovery procedure..."

# 1. Check system resources
if [ $(free -m | awk 'NR==2{printf "%.1f", $3*100/$2}') -gt 90 ]; then
    echo "Memory usage critical - restarting service"
    kubectl rollout restart deployment/wasm-torch
fi

# 2. Check disk space
if [ $(df / | tail -1 | awk '{print $5}' | sed 's/%//') -gt 85 ]; then
    echo "Disk space critical - cleaning logs"
    find /var/log/wasm-torch -name "*.log" -mtime +7 -delete
fi

# 3. Verify service health
if ! curl -f http://localhost:8000/health; then
    echo "Health check failed - scaling up"
    kubectl scale deployment wasm-torch --replicas=5
fi

echo "Recovery procedure completed"
```

## 📚 Best Practices

### Development
- Use feature flags for gradual rollouts
- Implement comprehensive integration tests
- Use semantic versioning for releases
- Maintain backward compatibility

### Deployment
- Always use blue-green or canary deployments
- Implement proper rollback mechanisms
- Monitor key business metrics during deployments
- Use infrastructure as code (Terraform, Pulumi)

### Operations
- Set up comprehensive alerting
- Implement automated incident response
- Regularly test disaster recovery procedures
- Maintain runbooks for common issues

### Security
- Regular security audits and penetration testing
- Keep dependencies updated
- Use secrets management (Vault, AWS Secrets Manager)
- Implement proper access controls

## 🔗 Additional Resources

- **API Documentation**: `/docs` endpoint when running
- **Prometheus Metrics**: `/metrics` endpoint
- **Health Checks**: `/health`, `/ready` endpoints
- **Admin Interface**: `/admin` (if enabled)
- **OpenAPI Schema**: `/openapi.json`

## 📞 Support

For production deployment support:
- **Issues**: https://github.com/yourusername/wasm-shim-for-torch/issues
- **Discussions**: https://github.com/yourusername/wasm-shim-for-torch/discussions
- **Documentation**: https://wasm-torch.readthedocs.io
- **Enterprise Support**: enterprise@wasm-torch.com

---

**Last Updated**: $(date)  
**Version**: 0.1.0  
**Status**: Production Ready ✅