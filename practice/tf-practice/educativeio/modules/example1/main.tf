provider "aws" {
  region = "us-east-1"
}

module "work_queue" {
  source     = "./sqs"
  queue_name = "work-queue"
}
