import pika
import json


# Connect to RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

# Declare exchange and queue
exchange_name = 'media_player_exchange'
channel.exchange_declare(exchange=exchange_name, exchange_type='fanout')
result = channel.queue_declare(queue='', exclusive=True)
queue_name = result.method.queue
channel.queue_bind(exchange=exchange_name, queue=queue_name)

def _playerlist() -> list:
    """Returns a list of all available media player services, for mediactl functions."""
    # Send message to get player list and wait for response
    channel.basic_publish(exchange=exchange_name, routing_key='', body='get_player_list')
    method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=True)
    return body.decode().split(',') if body else []

def playpause() -> int:
    """Toggles play/pause for all available media players, returns number successed."""
    players = _playerlist()
    worked = 0
    for player in players:
        try:
            # Send message to play/pause player and wait for response
            channel.basic_publish(exchange=exchange_name, routing_key=player, body='play_pause')
            method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=True)
            if body.decode() == 'success':
                worked += 1
        except:
            pass
    return worked

def next() -> int:
    """Go to next track for all available media players, returns number successed."""
    players = _playerlist()
    worked = 0
    for player in players:
        try:
            # Send message to play next track and wait for response
            channel.basic_publish(exchange=exchange_name, routing_key=player, body='next')
            method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=True)
            if body.decode() == 'success':
                worked += 1
        except:
            pass
    return worked

def prev() -> int:
    """Go to previous track for all available media players, returns number successed."""
    players = _playerlist()
    worked = 0
    for player in players:
        try:
            # Send message to play previous track and wait for response
            channel.basic_publish(exchange=exchange_name, routing_key=player, body='prev')
            method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=True)
            if body.decode() == 'success':
                worked += 1
        except:
            pass
    return worked

def stop() -> int:
    """Stop playback for all available media players, returns number successed."""
    players = _playerlist()
    worked = 0
    for player in players:
        try:
            # Send message to stop playback and wait for response
            channel.basic_publish(exchange=exchange_name, routing_key=player, body='stop')
            method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=True)
            if body.decode() == 'success':
                worked += 1
        except:
            pass
    return worked

def volumeup() -> int:
    """Increase volume for all available media players, returns number successed."""
    players = _playerlist()
    worked = len(players)
    for player in players:
        try:
            message = {'action': 'volumeup', 'player': player}
            channel.basic_publish(exchange='media_players', routing_key='', body=json.dumps(message))
        except:
            worked -= 1
    return worked

def volumedown() -> int:
    """Decrease volume for all available media players, returns number successed."""
    players = _playerlist()
    worked = len(players)
    for player in players:
        try:
            message = {'action': 'volumedown', 'player': player}
            channel.basic_publish(exchange='media_players', routing_key='', body=json.dumps(message))
        except:
            worked -= 1
    return worked

def status() -> list:
    """Returns list of dicts containing title, artist, & status for each media player."""
    players = _playerlist()
    details = []
    for player in players:
        try:
            message = {'action': 'status', 'player': player}
            channel.basic_publish(exchange='media_players', routing_key='', body=json.dumps(message))
            details.append(json.loads(channel.basic_get(queue='media_status')[2]))
        except:
            pass
    return details