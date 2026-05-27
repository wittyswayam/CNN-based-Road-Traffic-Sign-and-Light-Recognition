# ==============================================================================
# TrafficVision-AI :: Terraform Infrastructure
# ==============================================================================
# Provisions: EKS cluster, ElastiCache Redis, S3 model registry,
#             RDS Aurora (MLflow), ECR repository, IAM roles
#
# Usage:
#   cd infrastructure/terraform
#   terraform init
#   terraform plan -var-file=environments/production.tfvars
#   terraform apply
# ==============================================================================

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.27"
    }
  }

  backend "s3" {
    bucket         = "trafficvision-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "eu-west-1"
    encrypt        = true
    dynamodb_table = "trafficvision-terraform-lock"
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = "TrafficVision-AI"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# ── Variables ──────────────────────────────────────────────────────────────

variable "aws_region"      { default = "eu-west-1" }
variable "environment"     { default = "production" }
variable "cluster_name"    { default = "trafficvision-prod" }
variable "node_instance"   { default = "m5.xlarge" }
variable "gpu_instance"    { default = "g4dn.xlarge" }
variable "redis_node_type" { default = "cache.r6g.large" }
variable "db_instance"     { default = "db.r6g.large" }

# ── VPC ────────────────────────────────────────────────────────────────────

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.10.0/24", "10.0.11.0/24", "10.0.12.0/24"]
  public_subnets  = ["10.0.1.0/24",  "10.0.2.0/24",  "10.0.3.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = false     # HA: one per AZ

  tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# ── EKS Cluster ────────────────────────────────────────────────────────────

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.29"
  subnet_ids      = module.vpc.private_subnets
  vpc_id          = module.vpc.vpc_id

  cluster_endpoint_public_access = true

  eks_managed_node_groups = {
    api_nodes = {
      instance_types = [var.node_instance]
      min_size       = 3
      max_size       = 20
      desired_size   = 3
      labels = { role = "api" }
      taints = []
    }
    gpu_nodes = {
      instance_types = [var.gpu_instance]
      min_size       = 0
      max_size       = 4
      desired_size   = 0
      ami_type       = "AL2_x86_64_GPU"
      labels         = { role = "gpu-inference" }
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}

# ── ElastiCache Redis ──────────────────────────────────────────────────────

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "${var.cluster_name}-redis"
  description                = "TrafficVision inference cache"
  node_type                  = var.redis_node_type
  num_cache_clusters         = 3
  automatic_failover_enabled = true
  multi_az_enabled           = true
  engine_version             = "7.0"
  port                       = 6379
  subnet_group_name          = aws_elasticache_subnet_group.redis.name
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
}

resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.cluster_name}-redis-subnet"
  subnet_ids = module.vpc.private_subnets
}

# ── S3 Model Registry ──────────────────────────────────────────────────────

resource "aws_s3_bucket" "model_registry" {
  bucket = "trafficvision-model-registry-${var.environment}"
}

resource "aws_s3_bucket_versioning" "model_registry" {
  bucket = aws_s3_bucket.model_registry.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_lifecycle_configuration" "model_registry" {
  bucket = aws_s3_bucket.model_registry.id
  rule {
    id     = "archive-old-models"
    status = "Enabled"
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }
}

# ── ECR Repository ─────────────────────────────────────────────────────────

resource "aws_ecr_repository" "api" {
  name                 = "trafficvision-ai"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

# ── Outputs ────────────────────────────────────────────────────────────────

output "cluster_endpoint"    { value = module.eks.cluster_endpoint }
output "redis_endpoint"      { value = aws_elasticache_replication_group.redis.primary_endpoint_address }
output "model_bucket"        { value = aws_s3_bucket.model_registry.bucket }
output "ecr_repository_url"  { value = aws_ecr_repository.api.repository_url }
