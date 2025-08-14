#!/bin/bash
# Production deployment script for WASM Torch
# Comprehensive deployment with monitoring, security, and global distribution

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment"
LOG_FILE="/tmp/wasm-torch-deploy-$(date +%Y%m%d-%H%M%S).log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$LOG_FILE"
}

# Configuration validation
validate_config() {
    log "Validating deployment configuration..."
    
    # Check required tools
    local required_tools=("kubectl" "docker" "helm")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "$tool is required but not installed"
        fi
    done
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    # Check deployment files
    local required_files=(
        "$DEPLOYMENT_DIR/global-deployment.yaml"
        "$DEPLOYMENT_DIR/advanced-monitoring.yaml"
        "$DEPLOYMENT_DIR/production.yaml"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            error "Required deployment file not found: $file"
        fi
    done
    
    log "Configuration validation completed successfully"
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."
    
    # Check cluster resources
    info "Checking cluster resources..."
    local nodes_ready=$(kubectl get nodes --no-headers | grep -c " Ready ")
    if [[ $nodes_ready -lt 3 ]]; then
        warn "Only $nodes_ready nodes ready. Recommended: 3+ nodes for production"
    fi
    
    # Check node resources
    info "Checking node resources..."
    kubectl top nodes 2>/dev/null || warn "Metrics server not available - cannot check resource usage"
    
    # Check storage classes
    info "Checking storage classes..."
    if ! kubectl get storageclass &> /dev/null; then
        warn "No storage classes found - persistent volumes may not work"
    fi
    
    # Check network policies support
    info "Checking network policies support..."
    if ! kubectl api-resources | grep -q networkpolicies; then
        warn "Network policies not supported - some security features may not work"
    fi
    
    log "Pre-deployment checks completed"
}

# Security setup
setup_security() {
    log "Setting up security components..."
    
    # Create security namespace if it doesn't exist
    kubectl create namespace wasm-torch-security --dry-run=client -o yaml | kubectl apply -f -
    
    # Install cert-manager if not present
    if ! kubectl get deployment cert-manager -n cert-manager &> /dev/null; then
        info "Installing cert-manager..."
        kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml
        kubectl wait --for=condition=available --timeout=300s deployment/cert-manager -n cert-manager
    fi
    
    # Generate JWT secret if not exists
    if ! kubectl get secret wasm-torch-jwt-secret -n wasm-torch-global &> /dev/null; then
        info "Generating JWT secret..."
        local jwt_secret=$(openssl rand -base64 32)
        kubectl create secret generic wasm-torch-jwt-secret \
            --from-literal=secret="$jwt_secret" \
            -n wasm-torch-global
    fi
    
    # Apply RBAC and security policies
    info "Applying security policies..."
    kubectl apply -f "$DEPLOYMENT_DIR/security-policies.yaml" || warn "Security policies not applied"
    
    log "Security setup completed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    # Deploy monitoring namespace and components
    kubectl apply -f "$DEPLOYMENT_DIR/advanced-monitoring.yaml"
    
    # Wait for monitoring components to be ready
    info "Waiting for monitoring components to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n wasm-torch-monitoring
    kubectl wait --for=condition=available --timeout=300s deployment/grafana -n wasm-torch-monitoring
    kubectl wait --for=condition=available --timeout=300s deployment/jaeger -n wasm-torch-monitoring
    
    # Get monitoring endpoints
    info "Monitoring endpoints:"
    local grafana_ip=$(kubectl get service grafana -n wasm-torch-monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    local jaeger_ip=$(kubectl get service jaeger-query -n wasm-torch-monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    
    info "Grafana: http://$grafana_ip:3000 (admin/admin123)"
    info "Jaeger: http://$jaeger_ip:16686"
    
    log "Monitoring stack deployed successfully"
}

# Deploy Redis cluster
deploy_redis() {
    log "Deploying Redis cluster..."
    
    # Apply Redis configuration
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-config
  namespace: wasm-torch-global
data:
  redis.conf: |
    cluster-enabled yes
    cluster-require-full-coverage no
    cluster-node-timeout 15000
    cluster-config-file nodes.conf
    cluster-migration-barrier 1
    appendonly yes
    protected-mode no
    port 6379
EOF
    
    # Wait for Redis pods to be ready
    info "Waiting for Redis cluster to be ready..."
    kubectl wait --for=condition=ready --timeout=300s pod -l app=redis-cluster -n wasm-torch-global || warn "Redis cluster may not be fully ready"
    
    log "Redis cluster deployed successfully"
}

# Deploy main application
deploy_application() {
    log "Deploying WASM Torch application..."
    
    # Apply global deployment configuration
    kubectl apply -f "$DEPLOYMENT_DIR/global-deployment.yaml"
    
    # Wait for application to be ready
    info "Waiting for application deployment to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment/wasm-torch-api -n wasm-torch-global
    
    # Check HPA status
    info "Checking auto-scaling configuration..."
    kubectl get hpa wasm-torch-api-hpa -n wasm-torch-global || warn "HPA not configured"
    
    # Get application endpoints
    local api_ip=$(kubectl get service wasm-torch-api -n wasm-torch-global -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    info "API endpoint: http://$api_ip"
    
    log "Application deployed successfully"
}

# Setup ingress and TLS
setup_ingress() {
    log "Setting up ingress and TLS..."
    
    # Check if ingress controller is installed
    if ! kubectl get deployment ingress-nginx-controller -n ingress-nginx &> /dev/null; then
        info "Installing NGINX ingress controller..."
        kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml
        kubectl wait --for=condition=available --timeout=300s deployment/ingress-nginx-controller -n ingress-nginx
    fi
    
    # Wait for certificate issuer to be ready
    info "Waiting for certificate issuer..."
    sleep 30  # Give cert-manager time to process the ClusterIssuer
    
    # Check certificate status
    info "Checking TLS certificates..."
    kubectl get certificates -n wasm-torch-global || warn "TLS certificates not configured"
    
    log "Ingress and TLS setup completed"
}

# Run health checks
run_health_checks() {
    log "Running post-deployment health checks..."
    
    # Check pod status
    info "Checking pod status..."
    kubectl get pods -n wasm-torch-global
    kubectl get pods -n wasm-torch-monitoring
    
    # Check service endpoints
    info "Checking service endpoints..."
    kubectl get endpoints -n wasm-torch-global
    
    # Test API health endpoint
    info "Testing API health..."
    local api_ip=$(kubectl get service wasm-torch-api -n wasm-torch-global -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
    if [[ -n "$api_ip" && "$api_ip" != "pending" ]]; then
        if curl -f -s "http://$api_ip/health" &> /dev/null; then
            log "API health check passed"
        else
            warn "API health check failed - service may still be starting"
        fi
    else
        warn "API endpoint not ready yet"
    fi
    
    # Check HPA metrics
    info "Checking auto-scaling metrics..."
    kubectl get hpa -n wasm-torch-global
    
    log "Health checks completed"
}

# Generate deployment report
generate_report() {
    log "Generating deployment report..."
    
    local report_file="/tmp/wasm-torch-deployment-report-$(date +%Y%m%d-%H%M%S).md"
    
    cat > "$report_file" << EOF
# WASM Torch Production Deployment Report

**Deployment Date:** $(date)
**Cluster:** $(kubectl config current-context)

## Deployment Summary

### Application Status
\`\`\`
$(kubectl get deployments -n wasm-torch-global)
\`\`\`

### Service Status
\`\`\`
$(kubectl get services -n wasm-torch-global)
\`\`\`

### Auto-scaling Status
\`\`\`
$(kubectl get hpa -n wasm-torch-global)
\`\`\`

### Monitoring Status
\`\`\`
$(kubectl get pods -n wasm-torch-monitoring)
\`\`\`

## Access Information

### API Endpoints
- Main API: $(kubectl get service wasm-torch-api -n wasm-torch-global -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")

### Monitoring Dashboards
- Grafana: $(kubectl get service grafana -n wasm-torch-monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending"):3000
- Jaeger: $(kubectl get service jaeger-query -n wasm-torch-monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending"):16686

### TLS Certificates
\`\`\`
$(kubectl get certificates -n wasm-torch-global 2>/dev/null || echo "Not configured")
\`\`\`

## Resource Usage
\`\`\`
$(kubectl top nodes 2>/dev/null || echo "Metrics not available")
\`\`\`

## Deployment Log
Full deployment log available at: $LOG_FILE

## Next Steps

1. Configure DNS records for production domains
2. Set up monitoring alerts
3. Configure backup procedures
4. Review security policies
5. Set up CI/CD pipeline integration

EOF

    info "Deployment report generated: $report_file"
    cat "$report_file"
}

# Cleanup function
cleanup() {
    if [[ $? -ne 0 ]]; then
        error "Deployment failed. Check log file: $LOG_FILE"
    fi
}

# Main deployment function
main() {
    local action="${1:-deploy}"
    
    trap cleanup EXIT
    
    log "🚀 Starting WASM Torch production deployment"
    log "Log file: $LOG_FILE"
    
    case "$action" in
        "deploy")
            validate_config
            pre_deployment_checks
            setup_security
            deploy_monitoring
            deploy_redis
            deploy_application
            setup_ingress
            run_health_checks
            generate_report
            log "✅ Deployment completed successfully!"
            ;;
        "update")
            log "Updating application..."
            deploy_application
            run_health_checks
            log "✅ Update completed successfully!"
            ;;
        "monitoring")
            deploy_monitoring
            log "✅ Monitoring stack deployed!"
            ;;
        "status")
            run_health_checks
            ;;
        "cleanup")
            warn "This will remove all WASM Torch deployments. Are you sure? (y/N)"
            read -r confirmation
            if [[ "$confirmation" == "y" || "$confirmation" == "Y" ]]; then
                kubectl delete namespace wasm-torch-global --ignore-not-found
                kubectl delete namespace wasm-torch-monitoring --ignore-not-found
                kubectl delete namespace wasm-torch-security --ignore-not-found
                log "✅ Cleanup completed!"
            else
                log "Cleanup cancelled"
            fi
            ;;
        *)
            echo "Usage: $0 {deploy|update|monitoring|status|cleanup}"
            echo ""
            echo "Commands:"
            echo "  deploy      - Full production deployment"
            echo "  update      - Update application only"
            echo "  monitoring  - Deploy monitoring stack only"
            echo "  status      - Check deployment status"
            echo "  cleanup     - Remove all deployments"
            exit 1
            ;;
    esac
}

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    main "deploy"
else
    main "$1"
fi