from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status

from employees.models import Employee
from employees.serializers import EmployeeSerializer

import os
import shutil
import zipfile


class CreateEmployeeViewSet(viewsets.ModelViewSet):
    queryset = Employee.objects.all()
    serializer_class = EmployeeSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        idd = serializer.validated_data.get("employee_id")
        path = f"media/{idd}"

        uploaded_zip_file = request.FILES.get('images')
        main_image = request.FILES.get("main_image")

        if uploaded_zip_file is not None:
            zip_file_path = os.path.join(settings.MEDIA_ROOT, f'{idd}.zip')

            if not os.path.exists(os.path.dirname(zip_file_path)):
                os.makedirs(os.path.dirname(zip_file_path))

            with open(zip_file_path, 'wb+') as destination:
                for chunk in uploaded_zip_file.chunks():
                    destination.write(chunk)

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(path)
                for index, file in enumerate(os.listdir(path), start=1):
                    old_file_path = os.path.join(path, file)
                    new_file_path = os.path.join(path, f"{index}.jpg")
                    os.rename(old_file_path, new_file_path)

            os.remove(zip_file_path)

        if main_image is not None:
            main_image_path = os.path.join(path, 'main.jpg')
            with open(main_image_path, 'wb+') as destination:
                for chunk in main_image.chunks():
                    destination.write(chunk)

        serializer.validated_data["main_image"] = f"{idd}/main.jpg"
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class DeleteEmployeeViewSet(viewsets.ModelViewSet):
    queryset = Employee.objects.all()
    serializer_class = EmployeeSerializer

    def get_object(self):
        queryset = self.get_queryset()
        employee_id = self.kwargs['pk']
        obj = queryset.filter(employee_id=employee_id).first()
        self.check_object_permissions(self.request, obj)
        return obj

    def destroy(self, request, *args, **kwargs):
        pk = self.kwargs.get("pk")
        if "media" in os.listdir(os.getcwd()):
            try:
                user_id = self.queryset.get(employee_id=pk).employee_id
            except ObjectDoesNotExist:
                return Response(data={f"User does not exist"})
            for folder in os.listdir("media/"):
                if folder == user_id:
                    if os.path.isdir(os.path.join("media/", folder)):
                        shutil.rmtree(os.getcwd() + "\\media\\" + folder)

        return super().destroy(request, *args, *kwargs)


class ListEmployeeViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Employee.objects.all()
    serializer_class = EmployeeSerializer
