import json
import boto3
import email
import os
import numpy
from utils import one_hot_encode, vectorize_sequences


bucket_name = "hw4email"
ENDPOINT_NAME = 'sms-spam-classifier-mxnet-2020-05-07-23-49-36-008'
vocabulary_length = 9013
SOURCE_EMAIL = "jin@giveemail.com"

def extract_info(content):
    # From: "Andrew" <andrew@example.com>;
    # Date: Fri, 17 Dec 2010 14:26:21 -0800
    # Subject: Hello
    info = {}
    lines = content.split('\n')
    for line in lines:
        if line.startswith('From'):
            info['sender'] = line[line.find('<') + 1 : line.find('>')]
        elif line.startswith('Date'):
            info['date'] = line[line.find('Date') + 6: line.find('Date') + 22]
        elif line.startswith('Subject'):
            line = line.rstrip()
            info['subject'] = line[line.find('Subject') + 9: ]
            
    return info
    
    
def extract_body(content):
    body = ""
    if content.is_multipart():
        for part in content.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                body = part.get_payload(decode=True)  # decode
                break
    else:
        body = content.get_payload(decode=True)
    return body

def predict(body):
    runtime = boto3.client('sagemaker-runtime')

    test_messages = [body]
    one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                      Body=json.dumps(encoded_test_messages.tolist()),
                                      ContentType='application/json')

    responseBody = response['Body'].read().decode("utf-8")
    responseBody = json.loads(responseBody)
    return responseBody
    
    
    
def lambda_handler(event, context):
    s3 = boto3.client('s3')
    filename = event['Records'][0]['s3']['object']['key']
    print("filename: %s"%(filename))
    
    fileObj = s3.get_object(Bucket = bucket_name, Key = filename)
    file = fileObj['Body'].read().decode('utf-8')
    # print("file: %s"%(file.decode('utf-8')))
    
    content = email.message_from_string(file)
    info = extract_info(file)
    if info == {}:
        print("Cannot extract info from email header")
        return {
                'statusCode': 400,
                'body':'Error'
                }
    print("info: %s"%info)
    
    body = extract_body(content).decode("utf-8")
    print("email body: %s"%(body))
    
    result = predict(body)  # {'predicted_label':[[1.0]], 'predicted_probability':[[0.99]]} 
    
    
    label = "ham" if result['predicted_label'][0][0] == 0.0 else "spam"
    prob = round(result['predicted_probability'][0][0]*100, 2)
    reply = 'We received your email sent at {} with the subject {}.\r\n'\
            '\r\n'\
            'Here is a 240 character sample of the email body: \r\n'\
            '{}'\
            '\r\n'\
            'The email was categorized as {} with a '\
            '{}% confidence.'.format(info['date'], info['subject'], body[:240], label, prob)   
    print("reply: ", reply)
    ses = boto3.client('ses')
    response = ses.send_email(
                        Source=SOURCE_EMAIL,
                        Destination={
                            'ToAddresses': [
                                info['sender'],
                            ],
                            'CcAddresses': [],
                            'BccAddresses': []
                        },
                        Message={
                            'Subject': {
                                'Data': 'Automatic Reply',
                                'Charset': 'UTF-8'
                            },
                            'Body': {
                                'Text': {
                                    'Data': reply,
                                    'Charset': 'UTF-8'
                                }
                            }
                        })


    print("response: ", response)

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
