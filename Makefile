# This Makefile includes the main commands to build the API in AWS Lambda in order
# to be used in the production environment.
include .env

export $(shell sed 's/=.*//' .env)

connect-docker-ecr:
	aws ecr get-login-password --region ${CLOUD_REGION} | docker login --username AWS --password-stdin ${ECR_REPO}

build-lambda-img:
	docker build --platform linux/amd64 --provenance=false -t rag-backend-production -f src/backend_api/Dockerfile.lambda .

tag:
	docker tag rag-backend-production:latest ${ECR_REPO}/rag-backend-prod:latest

upload-lambda-img:
	docker push ${ECR_REPO}/rag-backend-prod:latest

infrastructure-plan:
	cd terraform && terraform init && terraform plan

infrastructure-apply:
	cd terraform && terraform apply

infrastructure-update:
	cd terraform && terraform apply -replace="aws_lambda_function.rag_backend_api"