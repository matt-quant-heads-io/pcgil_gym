import argparse
import pika
import json

parser = argparse.ArgumentParser(description='Starts distributed pod generations')

# write queue
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
write_channel = connection.channel()
write_channel.exchange_declare(exchange='pod_generation_exchange', exchange_type='direct')


if __name__ == '__main__':
    for level_set in range(1, 6):
        msg = json.dumps({'level_set': level_set})
        write_channel.basic_publish(exchange='pod_generation_exchange', routing_key="level_set_{}".format(level_set),
                                    body=msg)
