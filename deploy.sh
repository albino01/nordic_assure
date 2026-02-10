#!/bin/bash

## deploy.sh

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

#gcloud resource-manager tags bindings create \
#  --parent=//cloudresourcemanager.googleapis.com/projects/883165044435 \
#  --tag-value=tagValues/281481363704394
#done: true
#response:
#  '@type': type.googleapis.com/google.cloud.resourcemanager.v3.TagBinding
#  name: tagBindings/%2F%2Fcloudresourcemanager.googleapis.com%2Fprojects%2F883165044435/tagValues/281481363704394
#  parent: //cloudresourcemanager.googleapis.com/projects/883165044435
#  tagValue: tagValues/281481363704394
#  tagValueNamespacedName: nordic-assure/enviroment/development


#aolss113@CND1419FYF:nordic_assure$ gcloud resource-manager tags bindings list \
#  --parent=//cloudresourcemanager.googleapis.com/projects/883165044435
#---
#name: tagBindings/%2F%2Fcloudresourcemanager.googleapis.com%2Fprojects%2F883165044435/tagValues/281481363704394
#parent: //cloudresourcemanager.googleapis.com/projects/883165044435
#tagValue: tagValues/281481363704394
#tagValueNamespacedName: nordic-assure/enviroment/development

#
#./deploy.sh
# or
#gcloud run deploy nordic-assure-api --source . --region europe-west1 --allow-unauthenticated --min-instances 0 --max-instances 2
