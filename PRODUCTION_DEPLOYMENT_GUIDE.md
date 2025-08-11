# 🚀 WASM-Torch Production Deployment Guide

## Overview

WASM-Torch is a production-ready library that enables PyTorch models to run in WebAssembly environments with near-native performance. This guide covers complete production deployment including monitoring, security, scaling, and maintenance.

## 📋 Prerequisites

### System Requirements
- **CPU**: Multi-core x64 processor with SIMD support (recommended: 4+ cores)
- **Memory**: Minimum 8GB RAM (recommended: 16GB+)
- **Storage**: 50GB+ available space for models and cache
- **OS**: Linux (Ubuntu 20.04+), macOS 10.15+, or Windows 10+

### Software Dependencies
- **Docker**: 20.10+
- **Kubernetes**: 1.24+ (for cluster deployment)
- **Python**: 3.10+ (for development)
- **Node.js**: 18+ (for frontend integration)

## 🐳 Docker Deployment

### Quick Start with Docker Compose

1. **Clone the repository:**
```bash
git clone https://github.com/terragon-ai/wasm-torch.git
cd wasm-torch
```

2. **Configure environment:**
```bash
cp deployment/config/production.yaml.example deployment/config/production.yaml
# Edit configuration as needed
```

3. **Start the full stack:**
```bash
docker-compose -f deployment/docker-compose.production.yml up -d
```

4. **Verify deployment:**
```bash
curl http://localhost:8080/health
curl http://localhost:9090/metrics  # Prometheus metrics
```

### Production Docker Configuration

The production Docker setup includes:
- **API Server**: WASM-Torch API with auto-scaling
- **Nginx**: Reverse proxy with SSL termination
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Visualization dashboards
- **Redis**: Caching and session storage
- **Jaeger**: Distributed tracing
- **ELK Stack**: Centralized logging

### Environment Variables

```bash
# Core Configuration
PYTHONPATH=/app/src
LOG_LEVEL=INFO
WORKERS=4
CONFIG_PATH=/app/config/production.yaml

# Performance Tuning
WASM_TORCH_THREADS=4
WASM_TORCH_MEMORY_LIMIT_MB=512
WASM_TORCH_CACHE_SIZE_MB=256

# Security
WASM_TORCH_ENABLE_SECURITY=true
WASM_TORCH_MAX_MODEL_SIZE_MB=1000

# Monitoring
PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc
JAEGER_ENDPOINT=http://jaeger:14268/api/traces
```

## ☸️ Kubernetes Deployment

### Cluster Requirements

**Minimum cluster specifications:**
- 3 worker nodes
- 4 CPUs and 8GB RAM per node
- 100GB storage per node
- Network policy support
- Ingress controller (nginx recommended)

### Deploy to Kubernetes

1. **Create namespace and apply configurations:**
```bash
kubectl apply -f deployment/production.yaml
```

2. **Verify deployment:**
```bash
kubectl get pods -n wasm-torch
kubectl get services -n wasm-torch
kubectl get ingress -n wasm-torch
```

3. **Check auto-scaling:**
```bash
kubectl get hpa -n wasm-torch
kubectl describe hpa wasm-torch-hpa -n wasm-torch
```

### Scaling Configuration

**Horizontal Pod Autoscaler (HPA):**
- Min replicas: 3
- Max replicas: 20
- CPU threshold: 70%
- Memory threshold: 80%
- Custom metrics: inference_requests_per_second

**Vertical Pod Autoscaler (VPA):**
```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: wasm-torch-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: wasm-torch-api
  updatePolicy:
    updateMode: "Auto"
```

## 📊 Monitoring and Observability

### Metrics Collection

**Core Metrics:**
- Request latency (p50, p95, p99)
- Request rate (requests/second)
- Error rate (errors/total requests)
- Model inference time
- Memory and CPU usage
- Cache hit/miss rates

**Custom Metrics:**
```python
# Application metrics
INFERENCE_DURATION = Histogram('wasm_torch_inference_duration_seconds',
                              'Time spent on inference', ['model_type'])
MODEL_LOAD_COUNTER = Counter('wasm_torch_model_loads_total',
                           'Total model loads', ['model_name'])
CACHE_HIT_RATIO = Gauge('wasm_torch_cache_hit_ratio',
                       'Cache hit ratio')
```

### Grafana Dashboards

**Pre-configured dashboards include:**
1. **Overview Dashboard**: System health, request rates, error rates
2. **Performance Dashboard**: Latency percentiles, throughput metrics
3. **Resource Dashboard**: CPU, memory, disk usage
4. **Model Dashboard**: Model-specific metrics and performance
5. **Security Dashboard**: Security events and audit logs

### Alerting Rules

**Critical Alerts:**
- High error rate (>5% for 5 minutes)
- High latency (p95 >2s for 5 minutes)
- Pod crash looping
- High memory/CPU usage (>90%)
- Disk space critical (<10%)

**Warning Alerts:**
- Moderate error rate (>1% for 10 minutes)
- Elevated latency (p95 >1s for 10 minutes)
- Cache miss rate high (>50%)
- Queue depth growing

## 🔒 Security Configuration

### Network Security

**Ingress Configuration:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.wasm-torch.example.com
    secretName: wasm-torch-tls
```

**Network Policies:**
- Restrict ingress to specific namespaces
- Allow egress for monitoring and external APIs
- Block unnecessary inter-pod communication

### Application Security

**Input Validation:**
- Model size limits (configurable, default 1GB)
- File type validation (.wasm, .pth, .pt, .onnx)
- Input tensor validation (size, type, values)
- Request rate limiting

**Runtime Security:**
- Non-root container execution
- Read-only file systems where possible
- Security context constraints
- Regular security scanning

### Secrets Management

```bash
# Create secrets for production
kubectl create secret generic wasm-torch-secrets \
  --from-literal=model-encryption-key=$(openssl rand -base64 32) \
  --from-literal=monitoring-token=$(openssl rand -base64 16) \
  -n wasm-torch
```

## 🔧 Performance Tuning

### CPU and Memory Optimization

**CPU Settings:**
```yaml
resources:
  requests:
    cpu: 1
    memory: 2Gi
  limits:
    cpu: 4
    memory: 8Gi
```

**Memory Management:**
- Enable intelligent caching for frequently used models
- Configure memory pools for tensor operations
- Set appropriate garbage collection intervals
- Monitor for memory leaks

### WASM Compilation Optimization

**Compilation Flags:**
- `-O3`: Maximum optimization
- `-msimd128`: Enable SIMD operations
- `-mbulk-memory`: Bulk memory operations
- `--closure-compiler`: JavaScript minification

**Runtime Optimization:**
```yaml
wasm:
  optimization_level: "O3"
  enable_simd128: true
  initial_memory_mb: 16
  maximum_memory_mb: 512
```

### Batch Processing

**Optimal Batch Sizes:**
- Small models (< 10M params): Batch size 32-64
- Medium models (10-100M params): Batch size 8-16
- Large models (> 100M params): Batch size 1-4

**Adaptive Batching:**
```python
# Automatic batch size optimization
optimizer = get_advanced_optimizer()
optimal_batch = optimizer.optimize_batch_size(
    operation_type="inference",
    input_size=input_tensor.numel(),
    current_performance=last_inference_time
)
```

## 📈 Scaling Strategies

### Horizontal Scaling

**Load Balancing:**
- Round-robin for general requests
- Weighted routing for different model types
- Sticky sessions for stateful operations

**Multi-Region Deployment:**
```yaml
# Cross-region deployment with affinity
spec:
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            app: wasm-torch-api
        topologyKey: topology.kubernetes.io/zone
```

### Vertical Scaling

**Resource Requests:**
- Start conservative with 1 CPU, 2GB RAM
- Monitor and adjust based on usage patterns
- Use VPA for automatic resource adjustment

### Model-Specific Scaling

**Model Sharding:**
```python
# Distribute large models across multiple pods
class ModelShardingStrategy:
    def distribute_model(self, model, num_shards):
        # Split model layers across workers
        return self.create_shards(model, num_shards)
```

## 🔄 CI/CD Pipeline

### Automated Testing

```yaml
# .github/workflows/production-deploy.yml
name: Production Deployment
on:
  push:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Run comprehensive tests
      run: python run_comprehensive_tests.py
    
  security-scan:
    runs-on: ubuntu-latest
    steps:
    - name: Security vulnerability scan
      run: safety check -r requirements.txt
    
  deploy:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to production
      run: kubectl apply -f deployment/production.yaml
```

### Blue-Green Deployment

```bash
# Blue-green deployment script
#!/bin/bash
NAMESPACE="wasm-torch"
NEW_VERSION="v1.1.0"

# Deploy green environment
kubectl apply -f deployment/production-green.yaml -n $NAMESPACE

# Wait for green to be ready
kubectl rollout status deployment/wasm-torch-api-green -n $NAMESPACE

# Run health checks
kubectl exec deployment/wasm-torch-api-green -n $NAMESPACE -- python health-check.py

# Switch traffic
kubectl patch service wasm-torch-service -n $NAMESPACE -p '{"spec":{"selector":{"version":"'$NEW_VERSION'"}}}'

# Clean up blue environment
kubectl delete deployment wasm-torch-api-blue -n $NAMESPACE
```

## 🛠 Maintenance and Operations

### Regular Maintenance Tasks

**Daily:**
- Monitor system health and alerts
- Review application logs for errors
- Check resource utilization trends
- Verify backup completion

**Weekly:**
- Update security patches
- Review performance metrics
- Clean up old logs and cache
- Test disaster recovery procedures

**Monthly:**
- Performance optimization review
- Capacity planning assessment
- Security audit and penetration testing
- Documentation updates

### Backup and Recovery

**Data Backup:**
```bash
# Backup model cache and configuration
kubectl create job backup-$(date +%Y%m%d) \
  --from=cronjob/backup-job -n wasm-torch

# Backup persistent volumes
velero backup create wasm-torch-backup \
  --selector app=wasm-torch-api
```

**Disaster Recovery:**
```bash
# Restore from backup
velero restore create --from-backup wasm-torch-backup

# Verify restoration
kubectl get pods -n wasm-torch
python run_comprehensive_tests.py
```

### Log Management

**Log Rotation:**
```yaml
# Fluentd configuration for log rotation
<source>
  @type tail
  path /var/log/wasm-torch/*.log
  pos_file /var/log/fluentd/wasm-torch.log.pos
  tag wasm-torch.*
  rotate_wait 5
  refresh_interval 60
</source>
```

## 🚨 Troubleshooting Guide

### Common Issues

**High Memory Usage:**
1. Check for memory leaks in model loading
2. Adjust cache sizes in configuration
3. Enable memory profiling
4. Review garbage collection settings

**High Latency:**
1. Verify SIMD is enabled
2. Check batch size optimization
3. Review model compilation flags
4. Analyze network latency

**Pod Crashes:**
1. Check resource limits
2. Review application logs
3. Verify health check configuration
4. Check for OOM kills

### Debug Commands

```bash
# Pod debugging
kubectl logs -f deployment/wasm-torch-api -n wasm-torch
kubectl exec -it deployment/wasm-torch-api -n wasm-torch -- /bin/bash

# Performance debugging
kubectl top pods -n wasm-torch
kubectl describe hpa wasm-torch-hpa -n wasm-torch

# Network debugging
kubectl get networkpolicies -n wasm-torch
kubectl port-forward service/wasm-torch-service 8080:80 -n wasm-torch
```

## 📞 Support and Documentation

### Getting Help

- **Documentation**: [https://wasm-torch.readthedocs.io](https://wasm-torch.readthedocs.io)
- **GitHub Issues**: [https://github.com/terragon-ai/wasm-torch/issues](https://github.com/terragon-ai/wasm-torch/issues)
- **Community Forum**: [https://forum.wasm-torch.org](https://forum.wasm-torch.org)
- **Commercial Support**: [support@terragon-ai.com](mailto:support@terragon-ai.com)

### Additional Resources

- **API Documentation**: Generated automatically with OpenAPI/Swagger
- **Performance Benchmarks**: Regular benchmark results published
- **Security Advisories**: Subscribe to security update notifications
- **Best Practices Guide**: Community-maintained best practices

---

## 🎉 Congratulations!

Your WASM-Torch production deployment is now complete with:

✅ **High Availability**: Multi-pod deployment with auto-scaling  
✅ **Monitoring**: Comprehensive metrics, logs, and tracing  
✅ **Security**: Network policies, input validation, and secrets management  
✅ **Performance**: Optimized for production workloads  
✅ **Reliability**: Health checks, circuit breakers, and graceful degradation  
✅ **Observability**: Detailed dashboards and alerting

The system is ready to handle production PyTorch inference workloads in WebAssembly environments with enterprise-grade reliability and performance.