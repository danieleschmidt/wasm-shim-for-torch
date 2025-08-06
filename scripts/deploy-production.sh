#!/bin/bash

# Production Deployment Script for WASM Torch
# This script handles secure production deployment with proper validation

set -euo pipefail

# Configuration
NAMESPACE="wasm-torch"
APP_NAME="wasm-torch-app"
IMAGE_TAG="${1:-latest}"
DEPLOYMENT_ENV="${2:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Pre-deployment checks
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if kubectl is installed and configured
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
    fi
    
    # Check if we can connect to the cluster
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    # Check if docker is available for building
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
    fi
    
    # Check if required files exist
    local required_files=(
        "Dockerfile"
        "k8s-deployment.yml"
        ".env.production"
        "requirements.txt"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            error "Required file not found: $file"
        fi
    done
    
    log "Prerequisites check passed ✓"
}

# Build and tag the Docker image
build_image() {
    log "Building Docker image..."
    
    local image_name="wasm-torch:${IMAGE_TAG}"
    
    # Build the production image
    docker build --target production -t "$image_name" .
    
    # Run security scan on the image
    if command -v docker-scan &> /dev/null; then
        log "Running security scan on image..."
        docker scan "$image_name" || warn "Security scan completed with warnings"
    fi
    
    # Tag for registry if needed
    if [[ -n "${DOCKER_REGISTRY:-}" ]]; then
        local registry_image="${DOCKER_REGISTRY}/${image_name}"
        docker tag "$image_name" "$registry_image"
        log "Image tagged for registry: $registry_image"
    fi
    
    log "Docker image built successfully ✓"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply the deployment configuration
    kubectl apply -f k8s-deployment.yml
    
    # Update the image in the deployment
    kubectl set image "deployment/$APP_NAME" \
        wasm-torch="wasm-torch:${IMAGE_TAG}" \
        --namespace="$NAMESPACE"
    
    # Wait for rollout to complete
    kubectl rollout status "deployment/$APP_NAME" --namespace="$NAMESPACE" --timeout=600s
    
    log "Kubernetes deployment completed ✓"
}

# Verify deployment health
verify_deployment() {
    log "Verifying deployment health..."
    
    # Check if pods are ready
    local ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app=wasm-torch --field-selector=status.phase=Running --no-headers | wc -l)
    local total_pods=$(kubectl get pods -n "$NAMESPACE" -l app=wasm-torch --no-headers | wc -l)
    
    if [[ "$ready_pods" -eq 0 ]]; then
        error "No pods are ready. Deployment failed."
    fi
    
    log "Pods ready: $ready_pods/$total_pods"
    
    # Test health endpoint
    log "Testing health endpoint..."
    kubectl port-forward -n "$NAMESPACE" svc/wasm-torch-service 8080:80 &
    local port_forward_pid=$!
    
    # Wait a moment for port forwarding to establish
    sleep 5
    
    # Test the health endpoint
    if curl -f http://localhost:8080/health &> /dev/null; then
        log "Health check passed ✓"
    else
        warn "Health check failed - application may not be fully ready"
    fi
    
    # Clean up port forward
    kill $port_forward_pid || true
    
    log "Deployment verification completed ✓"
}

# Run post-deployment tasks
post_deployment_tasks() {
    log "Running post-deployment tasks..."
    
    # Display deployment status
    kubectl get all -n "$NAMESPACE"
    
    # Show recent logs
    log "Recent application logs:"
    kubectl logs -n "$NAMESPACE" -l app=wasm-torch --tail=20 --since=5m || true
    
    # Display service endpoints
    log "Service endpoints:"
    kubectl get services -n "$NAMESPACE"
    
    log "Post-deployment tasks completed ✓"
}

# Rollback function
rollback_deployment() {
    warn "Rolling back deployment..."
    kubectl rollout undo "deployment/$APP_NAME" --namespace="$NAMESPACE"
    kubectl rollout status "deployment/$APP_NAME" --namespace="$NAMESPACE"
    log "Rollback completed"
}

# Main deployment flow
main() {
    log "Starting production deployment for WASM Torch"
    log "Image tag: $IMAGE_TAG"
    log "Environment: $DEPLOYMENT_ENV"
    
    # Trap errors and attempt rollback
    trap 'error "Deployment failed. Consider running rollback if needed."' ERR
    
    check_prerequisites
    build_image
    deploy_to_kubernetes
    verify_deployment
    post_deployment_tasks
    
    log "🎉 Production deployment completed successfully!"
    log "Application is now available in the $NAMESPACE namespace"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback_deployment
        ;;
    "check")
        check_prerequisites
        ;;
    *)
        echo "Usage: $0 [deploy|rollback|check] [image-tag] [environment]"
        echo "  deploy   - Deploy the application (default)"
        echo "  rollback - Rollback to previous version"
        echo "  check    - Check prerequisites only"
        exit 1
        ;;
esac