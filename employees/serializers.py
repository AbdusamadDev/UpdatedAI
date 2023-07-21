from rest_framework import serializers
from employees.models import Employee


class EmployeeSerializer(serializers.ModelSerializer):
    images = serializers.FileField(required=True)
    main_image = serializers.ImageField(required=True)

    class Meta:
        model = Employee
        fields = '__all__'
