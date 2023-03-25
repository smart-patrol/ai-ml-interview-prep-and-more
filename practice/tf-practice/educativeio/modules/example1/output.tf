output "work_queue_name" {
  value = module.work_queue.queue
}

output "work_queue_dead_letter_name" {
  value = module.work_queue.dead_letter_queue
}
