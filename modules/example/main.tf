resource "google_storage_bucket" "bucket" {
  name = var.bucket_name
  location = var.region
  labels = var.tags
}
