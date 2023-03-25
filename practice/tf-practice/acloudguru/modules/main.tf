terraform {

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~>4.1"
    }
  }
}

provider "aws" {
  profile = "default"
  region  = "us-east-1"
}

module "ec2" {
  source = "./ec2/"
}




