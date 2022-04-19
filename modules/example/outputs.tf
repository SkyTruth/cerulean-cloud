# Output variable definitions
output "name" {
  description = "Name (id) of the bucket"
  value       = google_storage_bucket.storage_bucket.name
}

output "domain" {
  description = "Domain name of the bucket"
  value       = google_storage_bucket.storage_bucket.url
}