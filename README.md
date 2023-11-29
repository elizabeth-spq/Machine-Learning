<p align="center">
  <a href="https://api.faindet.codenauta.com">
    <img src="https://admin.faindet.faindet.com/assets/img/admin-logo.png" alt="Faindet">
  </a>
</p>
<p align="center">
    <em>Faindet is a web application used to hotel management</em>
</p>
## Get a list of all the outdated packages
<div class="termy">

```console
$ pip list --outdated

```

</div>
## Uninstall previous packages
<div class="termy">

```console
$ pip freeze > requirements_current.txt
$ pip uninstall -r requirements_current.txt

```

</div>
## Installation
<div class="termy">

```console
$ pip install -r requirements.txt
$ GET http://127.0.0.1:8000/bootstrap/
```

</div>
## DEPLOY VENV
<div class="termy">

```console
$ pyhton3 -m venv env
$ source env/bin/activate
$ deactivate
$ sudo apt-get install libpq-dev

```

</div>
## Run server
<div class="termy">

```console
$ uvicorn app.main:app --reload

```

</div>

## Compile on realtime and minify styles and scripts

<div class="termy">

```console
$ npm run dev

```

</div>

## Compile production and minify styles and scripts

<div class="termy">

```console
$ npm run build

```

</div>

## Run unit testing

<div class="termy">

```console
$ pytest tests/. -s

```

</div>
## Build image of container
<div class="termy">

```console
$ docker compose build

```

</div>
## Run the container
<div class="termy">

```console
$ docker compose up -d

```

</div>
## Delete the containers
<div class="termy">

```console
$ docker compose down

```

</div>
## Restore backup
<div class="termy">

```console
$ cat faindet.sql | docker exec -i faindet-db psql -U postgres -d faindet pg_restore -d faindet /app/db/faindet.sql
```

</div>
