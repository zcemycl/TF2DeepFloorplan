import os
from argparse import Namespace
from types import TracebackType
from typing import Any, Dict, Optional, Type

import pytest

import numpy as np
from flask.testing import FlaskClient
from pytest_mock import MockFixture

from dfp.app import app as create_app


class fakeMultiprocessing:
    def Pool(self):
        return self

    def map(self, *args: str, **kwargs: int) -> np.ndarray:
        return np.zeros([1, 32, 32, 3])

    def __enter__(self) -> object:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ):
        pass

    def __getitem__(self) -> np.ndarray:
        return np.zeros([1, 32, 32, 3])


@pytest.fixture
def client(mocker: MockFixture) -> FlaskClient:
    mp = fakeMultiprocessing()
    args = Namespace(
        image="resources/30939153.jpg",
        weight="",
        loadmethod="none",
        postprocess=False,
        colorize=False,
        save="/tmp/tmp.jpg",
    )
    content = Namespace(content=None)
    mocker.patch("dfp.app.mp.Pool", return_value=mp)
    mocker.patch("dfp.app.Namespace", return_value=args)
    mocker.patch("dfp.app.saveStreamFile", return_value=None)
    mocker.patch("dfp.app.saveStreamURI", return_value=None)
    mocker.patch("dfp.app.requests.get", return_value=content)
    mocker.patch("dfp.app.os.system", return_value=None)
    mocker.patch("dfp.app.mpimg.imsave", return_value=None)
    mocker.patch("dfp.app.send_file", return_value={"message": "success!"})
    return create_app.test_client()


def test_app_home(client: FlaskClient):
    resp = client.get("/")
    assert resp.status_code == 200
    assert isinstance(resp.json, dict)
    assert resp.json.get("message", "Hello Flask!")


def test_app_process_image(client: FlaskClient):
    resp = client.post("/process")
    assert resp.status_code == 400


def test_app_mock_process_empty(client: FlaskClient):
    headers: Dict[Any, Any] = {}
    data: Dict[Any, Any] = {}
    resp = client.post("/process", headers=headers, json=data)
    assert resp.status_code == 200
    assert resp.json.get("message", "success!")


def test_app_mock_process_uri(client: FlaskClient):
    headers: Dict[Any, Any] = {}
    data = {
        "uri": "",
        "postprocess": 1,
        "colorize": 1,
        "output": "/tmp",
    }
    resp = client.post("/process", headers=headers, json=data)
    os.system("rm *.jpg")
    assert resp.status_code == 200
    assert resp.json.get("message", "success!")


def test_app_mock_process_file(client: FlaskClient):
    files = {"file": (open("resources/30939153.jpg", "rb"), "30939153.jpg")}
    resp = client.post("/process", data=files)
    os.system("rm *.jpg")
    assert resp.status_code == 400
