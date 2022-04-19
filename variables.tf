variable google_region {
    type = string
    default = "europe-west1"
}


variable "override_bucket_name" {
    type = string  
}

locals {
  bucket_name = "${var.override_bucket_name != "" ? var.override_bucket_name : "bucket-${terraform.workspace}"}"
}
