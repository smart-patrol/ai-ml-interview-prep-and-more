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

resource "aws_instance" "app_server" {
  ami           = "ami-09d3b3274b6c5d4aa"
  instance_type = "t2.micro"
  subnet_id     = "subnet-0b9913888a49bf183"

}
