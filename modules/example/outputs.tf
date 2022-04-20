# Output variable definitions
output "name" {
  description = "Name (id) of the bucket"
  value       = google_storage_bucket.bucket.name
}

output "domain" {
  description = "Domain name of the bucket"
  value       = google_storage_bucket.bucket.url
}