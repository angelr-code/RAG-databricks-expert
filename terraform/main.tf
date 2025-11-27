# AWS Provider
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = "eu-west-1"
}


# Environment Variables injected into the Lambda function via Terraform (encrypted in AWS)
variable "qdrant_url" {
  description = "Qdrant Cloud URL"
  type        = string
  sensitive = true
}

variable "qdrant_api_key" {
  description = "Qdrant API Key"
  type        = string
  sensitive   = true
}

variable "openrouter_api_key" {
  description = "OpenRouter API Key"
  type        = string
  sensitive   = true
}

variable "backend_secret" {
  description = "Backend API access key"
  type = string
  sensitive = true
}

# Configuration variables for Qdrant and Embeddings
variable "qdrant_collection" {
  description = "Qdrant collection name"
  type        = string
  default     = "docs"
}

variable "embedding_dim" {
  description = "Embedding dimension"
  type        = number
  default     = 384
}

# ------------------------------------
# ECR Repository for Backend API Image
# ------------------------------------
resource "aws_ecr_repository" "api_repo" {
  name                 = "rag-backend-prod"
  image_tag_mutability = "MUTABLE"
  force_delete = true
  image_scanning_configuration {
    scan_on_push = true
  }
}

# ------------------------------------
# IAM Roles and Policies
# ------------------------------------

resource "aws_iam_role" "lambda_exec" {
  name = "rag_lambda_role_prod"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

# Permissions for CloudWatch Logs
resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}


# ------------------------------------
# AWS Lambda Resource
# ------------------------------------


resource "aws_lambda_function" "rag_backend_api" {
  function_name = "rag_backend_prod"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api_repo.repository_url}:latest"
  timeout       = 300
  memory_size = 1024

  environment {
    variables = {
      QDRANT_URL        = var.qdrant_url
      QDRANT_API_KEY    = var.qdrant_api_key
      QDRANT_COLLECTION = var.qdrant_collection
      EMBEDDING_DIM     = tostring(var.embedding_dim)
      OPENROUTER_API_KEY = var.openrouter_api_key
      BACKEND_SECRET = var.backend_secret
    }
  }
}

# ------------------------------------
# PPUBLIC ACCESS
# ------------------------------------

resource "aws_lambda_permission" "allow_public_access" {
  statement_id  = "AllowPublicFunctionUrlInvoke"
  action        = "lambda:InvokeFunctionUrl"
  function_name = aws_lambda_function.rag_backend_api.function_name
  principal     = "*"
  function_url_auth_type = "NONE"
}

# ------------------------------------
# Public URL
# ------------------------------------
resource "aws_lambda_function_url" "api_url" {
  function_name = aws_lambda_function.rag_backend_api.function_name
  authorization_type = "NONE" # Public URL 
  invoke_mode = "RESPONSE_STREAM" # Crucial for chat response streaming

  cors {
    allow_origins = ["*"]
    allow_methods = ["POST", "GET"]
    allow_headers = ["*"]
    max_age = 86400
  }
}


# ------------------------------------
# Outputs
# ------------------------------------
output "ecr_url" {
  value = aws_ecr_repository.api_repo.repository_url # Show ECR repo URL when finished
}

output "api_endpoint" {
  value = aws_lambda_function_url.api_url.function_url # Show the API URL
}
