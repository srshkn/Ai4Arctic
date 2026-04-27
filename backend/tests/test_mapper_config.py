from sqlalchemy.orm import configure_mappers


def test_orm_mappers_are_valid():
    configure_mappers()
