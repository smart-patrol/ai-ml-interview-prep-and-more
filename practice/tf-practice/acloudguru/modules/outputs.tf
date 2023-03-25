output "instance_id" {
  description = "ID of EC2 instance"
  value       = module.ec2.instance_id
}

output "instance_public_ip" {
  description = "Public IP address of the instance"
  value       = module.ec2.instance_public_ip
}
