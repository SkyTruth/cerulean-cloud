"""Generic auth module for fast API using hardcoded api key"""
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

api_keys = [os.getenv("API_KEY")]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")  # use token authentication


def api_key_auth(api_key: str = Depends(oauth2_scheme)):
    """api key auth dependency"""
    if api_key not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Forbidden"
        )
