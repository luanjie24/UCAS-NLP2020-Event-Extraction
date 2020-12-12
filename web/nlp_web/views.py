from django.shortcuts import render


def index(request):
    request.encoding = 'utf-8'
    context = {'input': request.POST['input'], 'output1': 'bala', 'output2': 'bala', 'output3': 'bala',
               'output4': 'bala', 'output5': 'bala'} if 'input' in request.POST and request.POST['input'] else {}
    return render(request, 'index.html', context)
