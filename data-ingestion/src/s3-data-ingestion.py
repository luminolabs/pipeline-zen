import boto3

def ingest_s3_data(bucket_name, object_key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=object_key)
    return obj['Body'].read()
