# Google cloud Cheatsheet

## Create a cloud machine with datalab
[Oficial page](https://cloud.google.com/datalab/docs/quickstarts/quickstart-gce-frontend)

### Previous:
#### Activate cloudresourcemanager
https://console.developers.google.com/apis/api/cloudresourcemanager.googleapis.com/overview?project=test-big-query-148315

#### Ativate Compute Engine API:
https://console.developers.google.com/apis/api/compute_component/overview?project=944447241305


### Instructions

```
gcloud compute networks create "datalab-gateway-network"  \
    --project "test-big-query-148315"   --description "Network for Datalab kern"

gcloud compute firewall-rules create datalab-gateway-network-allow-ssh \
    --project "test-big-query-148315"   --allow tcp:22   --network "datalab-gateway-network"   --description "Allow SSH access"

gsutil cp gs://cloud-datalab/gateway.yaml ./datalab-gateway.yaml

gcloud compute instances create "mymachine01" \
    --project "test-big-query-148315"   --zone "us-west1-a" \
    --network "datalab-gateway-network"   --image-family "container-vm" \
    --image-project "google-containers"   --metadata "google-container-manifest=$(cat datalab-gateway.yaml)"   --machine-type "n1-highmem-2"   --scopes "cloud-platform"

docker run -it -p "127.0.0.1:8081:8080" -v "${HOME}:/content" \
    -e "GATEWAY_VM=test-big-query-148315/us-west1-a/mymachine01" \
    gcr.io/cloud-datalab/datalab:local
```    
    
    