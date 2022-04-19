terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
      version = "4.10.0"
    }
    google = {
      source = "hashicorp/google"
      version = "4.18.0"
    }
  }

  backend "gcs" {
    bucket = "cerulean-cloud-state"
    prefix = "cerulean-cloud/"
  }
}

provider "aws" {
}

provider "google" {
  region = "europe-west1"
  zone   = "europe-west1-b"
}
