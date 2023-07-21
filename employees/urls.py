from django.urls import path
from .views import CreateEmployeeViewSet, DeleteEmployeeViewSet, ListEmployeeViewSet

app_name = 'employees'

urlpatterns = [
    path(
        'employees/create/',
        CreateEmployeeViewSet.as_view({'get': 'list', 'post': 'create'}),
        name='employee-list'
    ),
    path(
        'employees/<int:pk>/delete/',
        DeleteEmployeeViewSet.as_view(
            {'get': 'retrieve', 'put': 'update', 'delete': 'destroy'}
        ),
        name='employee-detail'
    ),
    path("employees/list/", ListEmployeeViewSet.as_view({"get": "list"}))
]
