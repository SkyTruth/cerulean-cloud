# Input variable definitions

variable "region" {
  description = "Name of the region."
  type        = string
}
variable "bucket_name" {
  description = "Name of the gcp bucket. Must be unique."
  type        = string
}

variable "tags" {
  description = "Tags to set on the bucket."
  type        = map(string)
  default     = {}
}