variable google_region {
    type = string
    default = "europe-west1"
}

variable google_project {
    type = string
    default = "cerulean-338116"
}


variable "bucket_name" {
    type = string 
    default = "a-bucket"
}

locals {
  bucket_name = "${var.google_project}-${terraform.workspace}-${var.bucket_name}"
}
