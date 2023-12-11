import boto3

def count_objects_in_bucket(s3_client, bucket_name):
    response = s3_client.list_objects(Bucket=bucket_name)
    return len(response.get('Contents', []))

def lambda_handler(event, context):
    environment_name:str="eb-cnn-trainer"
    elasticbeanstalk = boto3.client('elasticbeanstalk')
    s3_client = boto3.client('s3')
    batch_size:int=1000
    s3_bucket = event['Records'][0]['s3']['bucket']['name']
    num_objects = count_objects_in_bucket(s3_client, s3_bucket)

    if(num_objects>=batch_size):
        try:
            elasticbeanstalk.start_environment(EnvironmentName=environment_name)
        except:
            return {
                'statusCode': 500,
                'body': 'Elastic Beanstalk environment could not be started.'
            }
        else:
            return {
                'statusCode': 200,
                'body': 'Elastic Beanstalk environment started successfully.'
            }