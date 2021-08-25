'''
Parse commanline args
'''

def parseArgs(args):
    '''
    Parse command line arguments and organizes into a hash
    table (Python dictionary)

    parameters:
    args: list containing command line arguments, e.g:
          ['/items', 'item1', 'item2', 'item3']. Calling program
          passes in sys.argv[1:] (omitting the module name in sys.argv[0])

    returns:
          dictionary of the parameters, eg:
          {key_1: value, key_2: [value1, value2, ...]}

          If no named arguments ('/arg' or '-arg'), or before any named
          arguments, items are in the dictionary under the key 'None', e.g.,
          {'None': [arg1, arg2, ...]}

    effects:
          None
    '''
    count = len(args)
    params = {}
    key = None

    for i in range(count):
        if args[i][0] == '-' or args[i][0] == '/':
            key = args[i][1:]
            if params.get(key, None) is None:
                params[key] = None
        else:
            if key is None:
                key = 'None'
            current = params.get(key, None)
            if current is None:
                params[key] = args[i]
            elif isinstance(current, list):
                current.append(args[i])
                params[key] = current
            else:
                params[key] = [current] + [args[i]]

    return params
