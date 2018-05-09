# Instructions to work in AWS

## Create account
1.- Crearse una cuenta en AWS (hay que meter una tarjeta de crédito). Al crearse la cuenta se dispone de un año de ciertos servicios gratuitos. Se pueden levantar maquinas pequeñas gratis y probar a usar la plataforma casi sin coste (unos pocos centimos).

## Basic configuration
 Follow the next instructions https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/get-set-up-for-amazon-ec2.html
 - Sign Up for AWS
- Create an IAM User
- Create a Key Pair (download to local the key file)
- Create a Security Group. Open ports 22, 6006 and 8888. 


## Getting stated with EC2

Try to create an instance following the instructions here https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html?icmpid=docs_ec2_console
- Try to create a t2.micro: https://docs.aws.amazon.com/console/ec2/instances/connect/docs
    - Don't forget eliminate it.
- Create a budget alarm (to control the budget) http://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-alarms.html


## Create a spot instance with GPU for deep learning
- AMI: Select form the list below
- Instance type: p2.xlarge (aproximate cost of 0.3$/hour)
- EBS volume of 32Gb minimum.
- Key pair: The configured before
- Security group: The configured before


## AMI list

### AMI for deep learning with tensorflow 1.8:
- region: Ireland
- AMI id: ami-ced8edb7
- Name: sueiras-tensorflow-1.8

### ...

## Connect to AWS instances from windows using putty
https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html
https://portableapps.com/apps/internet/putty_portable

