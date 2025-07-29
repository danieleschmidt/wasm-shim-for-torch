# Deployment Guide

This guide covers deployment strategies, environment configuration, and operational considerations for WASM Shim for Torch.

## Deployment Overview

WASM Shim for Torch supports multiple deployment patterns, from simple static hosting to sophisticated CDN-based distribution with caching and optimization.

## Deployment Patterns

### 1. Static Hosting (Simplest)

Deploy compiled WASM modules and JavaScript files to any static hosting service:

```bash
# Build for production
make wasm
python -m build

# Deploy to static hosting
cp build/*.wasm dist/
cp build/*.js dist/
# Upload dist/ to your hosting service
```

**Suitable for**: Prototypes, small applications, development environments

**Requirements**: 
- Static file hosting
- HTTPS support (required for SharedArrayBuffer)
- COOP/COEP headers

### 2. CDN Distribution (Recommended)

Use Content Delivery Networks for optimal global performance:

```javascript
// Load from CDN
import WASMTorch from 'https://cdn.jsdelivr.net/npm/wasm-torch@latest/dist/wasm-torch.min.js';

// Initialize with CDN-hosted WASM
const runtime = await WASMTorch.init({
    wasmPath: 'https://cdn.jsdelivr.net/npm/wasm-torch@latest/dist/wasm-torch.wasm'
});
```

**Suitable for**: Production applications, global user base

**Benefits**:
- Global edge caching
- Reduced latency
- Automatic compression
- Version management

### 3. Enterprise Deployment

Self-hosted deployment with full control:

```yaml
# docker-compose.yml for production
version: '3.8'
services:
  wasm-torch-cdn:
    image: nginx:alpine
    volumes:
      - ./dist:/usr/share/nginx/html
      - ./nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "443:443"
      - "80:80"
    environment:
      - NGINX_HOST=your-domain.com
```

**Suitable for**: Enterprise environments, air-gapped deployments, compliance requirements

## Security Headers Configuration

### Required Headers

For threading support (SharedArrayBuffer), configure these headers:

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # Required for SharedArrayBuffer
    add_header Cross-Origin-Embedder-Policy require-corp;
    add_header Cross-Origin-Opener-Policy same-origin;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header Referrer-Policy strict-origin-when-cross-origin always;
    
    # CSP for WASM
    add_header Content-Security-Policy "
        default-src 'self';
        script-src 'self' 'unsafe-eval';
        worker-src 'self' blob:;
        wasm-src 'self';
    " always;
    
    location ~* \.(wasm)$ {
        add_header Content-Type application/wasm;
        add_header Cache-Control "public, max-age=31536000, immutable";
    }
}
```

### Apache Configuration

```apache
# .htaccess
Header always set Cross-Origin-Embedder-Policy require-corp
Header always set Cross-Origin-Opener-Policy same-origin

# WASM MIME type
<FilesMatch "\.wasm$">
    Header set Content-Type application/wasm
    Header set Cache-Control "public, max-age=31536000, immutable"
</FilesMatch>

# Compression
<IfModule mod_deflate.c>
    AddOutputFilterByType DEFLATE application/wasm
    AddOutputFilterByType DEFLATE application/javascript
</IfModule>
```

## Performance Optimization

### 1. Compression

Enable compression for all text-based assets:

```nginx
# Enable gzip compression
gzip on;
gzip_vary on;
gzip_min_length 1024;
gzip_types
    application/javascript
    application/json
    application/wasm
    text/css
    text/javascript
    text/plain;

# Brotli compression (if available)
brotli on;
brotli_types
    application/javascript
    application/json
    application/wasm
    text/css
    text/javascript;
```

### 2. Caching Strategy

Implement aggressive caching for immutable assets:

```nginx
# Long-term caching for versioned assets
location ~* \.(wasm|js)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
    add_header Vary "Accept-Encoding";
}

# Short-term caching for HTML
location ~* \.html$ {
    expires 1h;
    add_header Cache-Control "public, must-revalidate";
}
```

### 3. HTTP/2 Server Push

Optimize initial load times:

```nginx
# HTTP/2 Server Push for critical resources
location = /index.html {
    http2_push /wasm-torch.js;
    http2_push /model.wasm;
}
```

## Environment Configuration

### Development Environment

```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  dev-server:
    build:
      target: development
    ports:
      - "3000:3000"
    volumes:
      - .:/workspace
    environment:
      - NODE_ENV=development
      - WASM_TORCH_DEBUG=1
    command: npm run dev
```

### Staging Environment

```yaml
# docker-compose.staging.yml
version: '3.8'
services:
  staging-server:
    build:
      target: production
    ports:
      - "8080:80"
    environment:
      - NODE_ENV=staging
      - WASM_TORCH_DEBUG=0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Production Environment

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  production-server:
    image: wasm-torch:latest
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
    ports:
      - "443:443"
    environment:
      - NODE_ENV=production
      - ENABLE_MONITORING=true
    secrets:
      - ssl_certificate
      - ssl_private_key
```

## Monitoring and Observability

### 1. Application Metrics

```javascript
// Performance monitoring
class WASMTorchMetrics {
    constructor() {
        this.metrics = {
            modelLoadTime: new Map(),
            inferenceTime: new Map(),
            memoryUsage: new Map(),
            errorCount: 0
        };
    }
    
    recordModelLoad(modelName, loadTime) {
        this.metrics.modelLoadTime.set(modelName, loadTime);
        this.sendMetric('model_load_time', loadTime, { model: modelName });
    }
    
    recordInference(modelName, inferenceTime) {
        const times = this.metrics.inferenceTime.get(modelName) || [];
        times.push(inferenceTime);
        this.metrics.inferenceTime.set(modelName, times);
        
        this.sendMetric('inference_time', inferenceTime, { model: modelName });
    }
    
    recordError(error, context) {
        this.metrics.errorCount++;
        this.sendMetric('error_count', 1, { 
            error: error.name, 
            context: context 
        });
    }
    
    sendMetric(name, value, tags = {}) {
        // Send to your monitoring system
        if (window.analytics) {
            window.analytics.track('wasm_torch_metric', {
                metric: name,
                value: value,
                ...tags
            });
        }
    }
}
```

### 2. Infrastructure Monitoring

```yaml
# monitoring-stack.yml
version: '3.8'
services:
  prometheus:
    image: prometheus/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
      
  nginx-exporter:
    image: nginx/nginx-prometheus-exporter
    ports:
      - "9113:9113"
    command:
      - -nginx.scrape-uri=http://nginx:8080/nginx_status
```

### 3. Error Tracking

```javascript
// Error tracking integration
class ErrorTracker {
    constructor(options = {}) {
        this.endpoint = options.endpoint;
        this.apiKey = options.apiKey;
        this.environment = options.environment || 'production';
    }
    
    captureException(error, context = {}) {
        const errorData = {
            message: error.message,
            stack: error.stack,
            timestamp: new Date().toISOString(),
            environment: this.environment,
            userAgent: navigator.userAgent,
            url: window.location.href,
            context: context
        };
        
        // Send to error tracking service
        fetch(this.endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.apiKey}`
            },
            body: JSON.stringify(errorData)
        }).catch(console.error);
    }
}

// Global error handling
window.addEventListener('error', (event) => {
    errorTracker.captureException(event.error, {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno
    });
});

window.addEventListener('unhandledrejection', (event) => {
    errorTracker.captureException(new Error(event.reason), {
        type: 'unhandled_promise_rejection'
    });
});
```

## Deployment Automation

### 1. CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  release:
    types: [published]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
          npm install
          
      - name: Build WASM components
        run: |
          make setup-emscripten
          make wasm
          
      - name: Build Python package
        run: python -m build
        
      - name: Run security scan
        run: ./scripts/security-scan.sh
        
      - name: Build Docker images
        run: |
          docker build -t wasm-torch:${{ github.ref_name }} .
          docker build -t wasm-torch:latest .
          
      - name: Deploy to staging
        run: |
          docker-compose -f docker-compose.staging.yml up -d
          ./scripts/health-check.sh staging
          
      - name: Run integration tests
        run: pytest tests/integration/ -v
        
      - name: Deploy to production
        if: success()
        run: |
          docker-compose -f docker-compose.prod.yml up -d
          ./scripts/health-check.sh production
          
      - name: Notify deployment
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: 'WASM Torch deployment ${{ job.status }}'
```

### 2. Blue-Green Deployment

```bash
#!/bin/bash
# blue-green-deploy.sh

set -euo pipefail

NEW_VERSION="$1"
CURRENT_ENV=$(docker ps --format "table {{.Names}}" | grep -E "(blue|green)" | head -1 | sed 's/.*-\(blue\|green\).*/\1/')

if [[ "$CURRENT_ENV" == "blue" ]]; then
    NEW_ENV="green"
else
    NEW_ENV="blue"
fi

echo "Current environment: $CURRENT_ENV"
echo "Deploying to: $NEW_ENV"

# Deploy to new environment
docker-compose -f docker-compose.${NEW_ENV}.yml up -d

# Health check
if ./scripts/health-check.sh "$NEW_ENV"; then
    echo "Health check passed. Switching traffic..."
    
    # Switch load balancer
    ./scripts/switch-traffic.sh "$NEW_ENV"
    
    # Wait for traffic to drain
    sleep 30
    
    # Stop old environment
    docker-compose -f docker-compose.${CURRENT_ENV}.yml down
    
    echo "Deployment successful!"
else
    echo "Health check failed. Rolling back..."
    docker-compose -f docker-compose.${NEW_ENV}.yml down
    exit 1
fi
```

## Scaling Considerations

### 1. Horizontal Scaling

```yaml
# kubernetes-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wasm-torch-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: wasm-torch
  template:
    metadata:
      labels:
        app: wasm-torch
    spec:
      containers:
      - name: wasm-torch
        image: wasm-torch:latest
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: wasm-torch-service
spec:
  selector:
    app: wasm-torch
  ports:
  - protocol: TCP
    port: 80
    targetPort: 80
  type: LoadBalancer
```

### 2. Auto-scaling

```yaml
# horizontal-pod-autoscaler.yml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: wasm-torch-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: wasm-torch-deployment
  minReplicas: 2
  maxReplicas: 10
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

## Disaster Recovery

### 1. Backup Strategy

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup application code
tar -czf "$BACKUP_DIR/app.tar.gz" /app

# Backup configuration
cp -r /etc/nginx "$BACKUP_DIR/"
cp -r /etc/ssl "$BACKUP_DIR/"

# Backup metrics and logs
tar -czf "$BACKUP_DIR/metrics.tar.gz" /var/log/nginx /var/log/app

echo "Backup completed: $BACKUP_DIR"
```

### 2. Recovery Procedures

```bash
#!/bin/bash
# recover.sh

BACKUP_DIR="$1"

if [[ ! -d "$BACKUP_DIR" ]]; then
    echo "Backup directory not found: $BACKUP_DIR"
    exit 1
fi

echo "Starting recovery from: $BACKUP_DIR"

# Stop services
docker-compose down

# Restore application
tar -xzf "$BACKUP_DIR/app.tar.gz" -C /

# Restore configuration
cp -r "$BACKUP_DIR/nginx" /etc/
cp -r "$BACKUP_DIR/ssl" /etc/

# Start services
docker-compose up -d

# Verify recovery
./scripts/health-check.sh production

echo "Recovery completed"
```

## Security Considerations

### 1. Access Control

```nginx
# IP-based access control
location /admin {
    allow 192.168.1.0/24;
    allow 10.0.0.0/8;
    deny all;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
location /api {
    limit_req zone=api burst=20 nodelay;
}
```

### 2. SSL/TLS Configuration

```nginx
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 1d;
ssl_session_tickets off;

# OCSP stapling
ssl_stapling on;
ssl_stapling_verify on;
```

## Troubleshooting

### Common Deployment Issues

1. **COOP/COEP Headers Missing**:
   ```bash
   # Check headers
   curl -I https://your-domain.com/app.html
   
   # Should include:
   # cross-origin-embedder-policy: require-corp
   # cross-origin-opener-policy: same-origin
   ```

2. **WASM Loading Failures**:
   ```javascript
   // Debug WASM loading
   try {
       const module = await WebAssembly.instantiateStreaming(
           fetch('model.wasm')
       );
   } catch (error) {
       console.error('WASM loading failed:', error);
   }
   ```

3. **Performance Issues**:
   ```bash
   # Monitor resource usage
   docker stats wasm-torch-container
   
   # Check memory usage
   curl http://localhost/metrics
   ```

This deployment guide provides comprehensive coverage of production deployment scenarios for WASM Shim for Torch, from simple static hosting to enterprise-grade Kubernetes deployments.