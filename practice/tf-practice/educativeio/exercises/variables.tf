variable "region" {
  type        = string
  default     = "us-east-1"
  description = "region to launch in "
}
variable "main_vpc_cidr" {}
variable "public_subnets" {}
variable "private_subnets" {}
