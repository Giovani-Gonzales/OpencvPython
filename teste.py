import time
from comtypes.client import CreateObject

class PyE3DataAccess(object):
    def __init__(self, server='localhost'):
        super(PyE3DataAccess, self).__init__()
        self._engine = CreateObject("{80327130-FFDB-4506-B160-B9F8DB32DFB2}")
        self._engine.Server = server

    def lerValorE3(self, pathname):
        return self._engine.ReadValue(pathname)
    
    def escreverValorE3(self, pathname, date, quality, value):
        return self._engine.WriteValue(pathname, date, quality, value)
    
if __name__ == '__main__':

    # Identificação para a conexão com o elipse
    pyE3DataAccess = PyE3DataAccess(server="localhost")
    caminhoTexto = 'Dados.abc1234.value'
    date = time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime())

    valorTag = input('Escreva um valor para escrever na tag: ')
    
    pyE3DataAccess.escreverValorE3(pathname=caminhoTexto, date=date, quality=192, value = valorTag)