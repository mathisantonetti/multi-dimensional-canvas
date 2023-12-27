import numpy as np

def oneclose(e, L, eps):
    '''
    Parameters :
    _ L : list of vectors (in matrix shape)
    _ e : element possibly near to L
    _ eps : maximum accuracy tolerated of e with one element in L

    Returns : boolean
    if e is eps-close to L, returns true
    else returns false
    '''
    n = L.shape[0]
    if(n == 0):
        return false

    for i in range(n):
        if(norm(e - L[i, :]) > eps):
            return false
    return true

def updateVecInBase(i, base):
    '''
    Parameters :
    _ i : integer vector that symbolize the number to update
    _ limits : list of integer 2D-vectors to symbolize the box of the base

    Returns :
    The number i+1 in the base notation
    '''
    d = len(i)
    new_i = i
    for k in range(d):
        if(new_i[k] <= base[k][2] - 1):
            new_i[k] += 1
            break
        elif(k <= d - 1):
            new_i[k] = base[k][1]
        else:
            return -1
    return new_i

def ordered_insert(list, elementToAdd, Compare_list, compared_element):
    '''
    Parameters :
    _ list : list of elements sorted with the order defined by Compare_list
    _ elemntsToAdd : the element to add to list if compared_element greater than one in list.
    _ Compare_list : list of real numbers defining the order of list
    _ compared_element : real number defining the order

    Returns :
    list updated with elementToAdd (added to the right place defined by its order)
    '''
    if(len(Compare_list) != list.shape[0]):
        print("\n list : ", list)
        print("\n clist : ", Compare_list)
    K = len(Compare_list)
    new_list = np.copy(list)
    new_compare_list = np.copy(Compare_list)
    dim = len(elementToAdd)
    for k in range(K-1):
        if(Compare_list[k] < compared_element and compared_element <= Compare_list[k+1]):
            Listcat = np.concatenate((list[2:k, :], np.reshape(elementToAdd, 1, dim)), dims=1) # add the element
            new_list = np.concatenate((Listcat, list[(k+1):K, :]), dims=1)
            Comparecat = np.concatenate((Compare_list[2:k], [compared_element]), dims=1)
            new_compare_list = np.concatenate((Comparecat, Compare_list[(k+1):K]), dims=1)
            break

    if(Compare_list[K] < compared_element):
        new_list[1:(K-1), :] = np.copy(list[2:K, :])
        new_compare_list[1:(K-1)] = np.copy(Compare_list[2:K])
        new_list[K, :] = elementToAdd
        new_compare_list[K] = compared_element
    return new_list, new_compare_list

def decompo_base(n, d, base=10):
    '''
    Parameters :
    _ n : Number to return
    _ d : return format control
    _ base : base control

    Returns :
    [a1, ..., ad] such that n = a1 + a2*base + ... + ad*base^(d-1) (base decomposition)
    '''
    res = np.zeros(d, dtype=int)
    x0 = n
    for i in range(d):
        q = int(x0/base**(d-i))
        res[i] = q
        x0 = x0 - q*base**(d-i)
    return res

def div(a,b):
    if(np.abs(b) > 0):
        return a/b
    else:
        return np.sign(a)*np.Inf