terraform {
  backend "s3" {
    bucket         = "cjf-terraform-state-bucket-123"
    region         = "us-east-1"
    key            = "backend.tfstate"
    dynamodb_table = "terraformstatelock"
  }
}
