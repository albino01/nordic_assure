#!/bin/bash

# deploy.sh

gcloud config set project nordic-assure
gcloud billing projects link nordic-assure --billing-account 01166E-9A9F0B-8FC7C5
gcloud config set run/region europe-west1
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

gcloud run deploy nordic-assure-api \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --cpu 1 \
  --memory 1Gi \
  --min-instances 0 \
  --max-instances 2