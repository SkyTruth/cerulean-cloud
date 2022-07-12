"""cloud function scene relevancy handler"""

import asyncio
import os

import asyncpg


async def get_row():
    """get a row"""
    conn = await asyncpg.connect(os.getenv("DB_URL"))
    row = await conn.fetchrow("SELECT * FROM trigger")
    return row


def main(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    row = loop.run_until_complete(get_row())
    print(row)
    request_json = request.get_json()
    if request.args and "message" in request.args:
        return request.args.get("message")
    elif request_json and "message" in request_json:
        return request_json["message"]
    else:
        return "Hello World!"
