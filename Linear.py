"""Goal is to implement Linear Algebra stuff, start with basics, move up"""
import numpy as np #arrays are more efficient than python lists # guess I didn't listen to my own advice lol
import math

def gcf(n1, n2):
    """Greatest common factor using euclid algorithm"""
    if n2>n1: n1,n2 = n2,n1 # swap if not in right order, should be n1 > n2
    while(True):
        r = n1%n2
        if r==0:
            return n2
        n1 = n2
        n2 = r

def lcm(num1, num2):
    """Least common multiple dependent on gcm algorithm"""
    return int(num1*num2/gcf(num1,num2))


    


class Matrix:
    def __init__(self, rows, cols, data=None):
        """Init magic method lol"""
        #if rows==cols: SquareMatrix.__init__(rows,data) # would this cause problems? Or at least it might not extend functionality
            # HOW TO DO THIS??? Sus even
        self.rows = rows
        self.cols = cols
        if data is not None:
            self.data = data #better way to do this?
        else:
            self.data = [[] for _ in range(rows)]
            # self.data should be arr[rows][cols]
            
    @classmethod
    def vectorToMatrix(cls, vector):# TODO: check if this works
        if not isinstance(vector,Vector): raise TypeError(f"{vector} should be a Vector object")
        return cls(vector.rows,1,[[val] for val in vector])
    
    @classmethod
    def colsToMatrix(cls, basis):
        
        if any(not isinstance(col,Vector) for col in basis): raise TypeError("Basis must be a set of vector objects")
        
        ogCol = None
        for col in basis:
            ogCol = col
            break
        if not ogCol: raise ValueError("You would create a null matrix or some shit, nuh uh") 
        
        if any(len(col) != len(ogCol) for col in basis): raise ValueError("All vectors must be of same length")
        
        ret = cls(len(ogCol),len(basis))
        
        
        for col in basis:
            for i in range(len(ogCol)):
                ret.data[i].append(col[i])
            
        return ret        

    def __len__(self):
        """Returns size of matrix, ie total # of entries"""
        return self.rows*self.cols # is this the right way to implement this? Is there a better way?
    
    
    def __hash__(self):
        return hash(((self.data[i][j] for i in self.rows)for j in range(self.cols)))
    
    def __eq__(self,other):
        """If two matrices are equal or not"""
        if not isinstance(other,Matrix): return False # add edge case for vectors?
        if self.rows!=other.rows or self.cols!=other.cols: return False            
        return all (self.data[i][j]==other.data[i][j] for i in range(self.rows) for j in range(self.cols))
    
    def __str__(self):  
        """Gives a nice lil thing"""
        ret = "" 
        for row in self.data:
            r = []
            for entry in row:
                if round(entry,6)==int(entry): 
                    r.append(int(entry))
                else:
                    r.append(round(entry,3))
            ret += f'|{str(r)[1:-1]}|\n'
        return ret

    def __repr__(self): return str(self.data)
    
    def getCol(self,num):
        """Returns num'th column (idx starts at 0) as a vector """
        if num<0 or num>= self.cols: raise ValueError(f"Cannot retrieve column {num} of given Matrix; out of bounds error")
        return Vector(*[self.data[val][num] for val in range(self.rows)])
    
    def getRow(self,num):#TODO: test this
        """Retursn num'th row (idx starts at 0) as a vector"""
        if num<0 or num >= self.rows: raise ValueError(f"Cannot retrieve row {num} of given Matrix; out of bounds error")
        return Vector(*self.data[num][:]).transpose()#should I transpose?
    
    def allRows(self):#TODO: implement
        """Returns set of all row vectors in self"""
        pass
    
    def allCols(self):#TODO: implement
        """Returns set of all column vectors in matrix self"""
        pass
    
    
    
    def getPivots(self):
        """Returns list of leading entry coordinates of the matrix if there are any"""
        leadings = []
        for i in range(self.rows):
            for j in range(self.cols):
                if round(self.data[i][j],5) !=0: 
                    leadings.append((i,j))
                    break
        return leadings
    
    def interchange(self, row1, row2): # elementary row operation
        """Elementary row operation; swap two rows"""
        self.data[row1],self.data[row2] = self.data[row2],self.data[row1]

    def scale(self,row,scale):
        """Elementary row operation; multiply row by a non-zero constant
        will replace row with row*scale"""
        if scale==0: raise ValueError("Cannot scale row by 0")
        self.data[row] = [entry*scale for entry in self.data[row]]

    def replace(self,row1,scale1,row2,scale2):
        """Elementary row operation; replace row with sum of scalar multiple of itself and a scalar multiple of another row
        Will replace row1 with row1*scale1 + row2+scale2"""
        if scale1==0: raise ValueError("Cannot scale initial row by 0")
        self.data[row1] = [self.data[row1][i]*scale1 + self.data[row2][i]*scale2 for i in range(self.cols)]

    
    def isConsistent(self):
        """Returns boolean if consistent"""
        # if last pivot is in the final col, then false else true
        return self.rref().getPivots()[-1][1]!=self.cols-1



    
    def isRef(self):
        """Returns True/False if matrix is in row echelon form
        in Ref iff: 
        - all nonzero rows above any zero rows
        - each leading entry (leftmost nonzero entry in row) is to the right of leading entry above it
        - entries below leading entry are zero
        """
        #criteria 1: all nonzero rows above any zero rows
        for idx in range(self.rows-1): # don't care about last row
            if all(self.data[idx][i]==0 for i in range(self.cols)): # if there is a full row of zeros
                if any(self.data[j][i]!=0 for j in range(idx,self.rows) for i in range(self.cols)): 
                    return False
                else:
                    break# we checked that there are no problems with zero rows above nonzero rows
        
        #criteria 2: each leading entry (leftmost nonzero entry in row) is to the right of leading entry above it
                
        leadings = self.getPivots()
        if any(leadings[i][1] >= leadings[i+1][1] for i in range(len(leadings)-1)):
            return False
                
        # criteria 3: entries below leading entry are zero

        if any(self.data[idx][j]!=0  for (i,j) in leadings for idx in range(i+1,self.rows)): return False #implied by criteria 2?


        return True
        

    def isRref(self):
        """returns True/False if matrix is in reduced row echelon form
        In rref form iff:
        - in ref
        - leading entries all 1
        - each leading entry is the only nonzero entry in its column"""
        if not self.isRef(): return False # condition 1

        leadings  = self.getPivots()
        if any(self.data[i][j]!=1 for i,j in leadings): return False # condition 2

        if any(self.data[idx][j]!=0 for i,j in leadings for idx in range(i+1,self.rows) ): return False

        return True
    
    def ref(self):
        """Returns row equivalent matrix in row echelon form, plus list of row operations done"""
        L = [] # list of all row operations
        if self.isRef(): return self,L
        mat = Matrix(self.rows,self.cols,self.data[:][:]) # creates copy of data for manipulation


        currCol = 0
        currRow = 0 # is this always the same

        for _ in range(mat.rows): # stops infinite loops
            if mat.isRef(): break

            if any(mat.data[i][currCol]!=0 for i in range(currRow, mat.rows)):
                nonZero = 0
                for j in range(currRow,mat.rows):
                    if mat.data[j][currCol]!=0:
                        nonZero = j
                        if currRow!=nonZero:
                            L.append(f"interchange({currRow},{nonZero})")
                            mat.interchange(currRow,nonZero) # makes sure that nonzero row on top
                        break  # this should only break out of the for loop
                for idx in range(currRow+1,mat.rows):
                    if mat.data[currRow][currCol]==0: continue
                    factor = mat.data[idx][currCol] / mat.data[currRow][currCol]
                    
                    L.append(f"replace({idx},1,{currRow},{round(-factor,5)})")
                    
                    mat.replace(idx,1,currRow,-factor) # should make everything zero

                currRow +=1 
                
            currCol +=1 #done regardless

        return mat, L


    def _rref(self):
        """Returns matrix that is in reduced row echelon form using row reduction algorithm
        while(matrix is not in ref):
            begin w/ left most nonzero col, ka pivot col. pivot position
            select nonzero entry in pivot col; if needed, interchange two rows to move this to pivot position
            use row replacement to create 0s below the pivot
        start at rightmost pivot + work upward and to the left, create 0s above each pivot
        if pivot is not 1, scale so it is """
        L = [] # list of all row operations
        if self.isRref(): return self,L
        mat = Matrix(self.rows,self.cols,self.data[:][:]) # creates copy of data for manipulation


        currCol = 0
        currRow = 0 # is this always the same

        for _ in range(mat.rows): # stops infinite loops
            if mat.isRef(): break

            if any(mat.data[i][currCol]!=0 for i in range(currRow, mat.rows)):
                nonZero = 0
                for j in range(currRow,mat.rows):
                    if mat.data[j][currCol]!=0:
                        nonZero = j
                        if currRow!=nonZero:
                            L.append(f"interchange({currRow},{nonZero})")
                            mat.interchange(currRow,nonZero) # makes sure that nonzero row on top
                        break  # this should only break out of the for loop
                for idx in range(currRow+1,mat.rows):
                    if mat.data[currRow][currCol]==0: continue
                    factor = mat.data[idx][currCol] / mat.data[currRow][currCol]
                    
                    L.append(f"replace({idx},1,{currRow},{round(-factor,5)})")
                    
                    mat.replace(idx,1,currRow,-factor) # should make everything zero

                currRow +=1 
                
            currCol +=1 #done regardless
        
                    
            #add something to make sure zero rows are on the bottom if detected; already accounted for tho??
        # now in REF!
        leadings = mat.getPivots()[::-1]
        for leading in leadings:
            for idx in range(leading[0]): #row of leading entry
                factor = mat.data[idx][leading[1]] / mat.data[leading[0]][leading[1]]
                L.append(f"replace({idx},1,{leading[0]},{round(-factor,5)})")
                mat.replace(idx,1,leading[0],-factor)
                
            scalar = 1/mat.data[leading[0]][leading[1]]
            if scalar !=1:
                L.append(f"scale({leading[0]},{round(scalar,5)})")
                mat.scale(leading[0],scalar) #scale everything down

        #check for it working on augmented matrices?

        return mat,L
    
    
    def rref(self):
        """Returns RREF form of matrix"""
        return self._rref()[0]
    
    def opsToRref(self):
        """Returns operations used to get from self to RREF form"""
        return self._rref()[1]
    
    
    def __add__(self,other):
        if not isinstance(other,Matrix): raise TypeError("You fucking wanker!")
        if self.rows != other.rows or self.cols != other.cols: raise TypeError(f"Cannot add {self.rows}x{self.cols} matrix to a {other.rows}x{other.cols} matrix")
        ret = Matrix(self.rows,self.cols,[[] for _ in range(self.rows)])
        for row in range(self.rows):
            for col in range(self.cols):
                ret.data[row].append( self.data[row][col] + other.data[row][col])
        return ret
    
    def __sub__(self,other):
        return self + (-other)
    
    def __pos__(self): return self
    
    def __neg__(self):
        return Matrix(self.rows,self.cols,[[-self.data[row][col] for row in range(self.rows)] for col in range(self.cols)])
    
    def __iter__(self):
        """Iterates through all values, going row-wise"""
        for arr in self.data:
            yield from arr
            
    def __rmul__(self,other):
        if  isinstance(other,int) or isinstance(other,float): return self * other
        raise ValueError("Cannot multiply matrix by all that (tongue out)")
         
    
    def __mul__(self,other):
        if isinstance(other,int) or isinstance(other,float):
            ret = Matrix(self.rows,self.cols)
            for row in range(self.rows):
                for col in range(self.cols):
                    ret.data[row].append( other * self.data[row][col])
            return ret
        if not (issubclass(type(other),Matrix)): raise TypeError("You fucking wanker! Can't multiply matrix with that ting")
        #implement stuff for vectors since they are a subset of matrices

        if self.cols != other.rows: raise ValueError(f"Cannot multiply a {self.rows}x{self.cols} matrix with a {other.rows}x{other.cols} matrix!")
        ret = Matrix(self.rows,other.cols)
        for col in range(ret.cols):
            for row in range(ret.rows):
                val = sum(self.data[row][i]*other.data[i][col] for i in range(self.cols))
                ret.data[row].append(val)
        return ret
    
    def __pow__(self,val):
        if val==1: return self
        else: raise ValueError("Cannot take non-square matrix to a power other than 1")
    
    def transpose(self):
        """Transposes matrix in O(n*m) time --> is there a more effecient way?"""
        ret = Matrix(self.cols,self.rows)
        for col in range(self.cols):
            for row in range(self.rows):
                ret.data[col].append(self.data[row][col])
        return ret
    
    def LuFactorization(self):
        """Returns two matrices: the lower triangle and the upper triangle"""
        lower = Matrix(self.rows,self.rows,IdentityMatrix(self.rows).data) # this should work
        
        upper = Matrix(self.rows,self.cols,self.data[:][:])
        L = [] # list of all row operations
        #Okay maybe I should create a Row Operation Object!
        #For now, we will have a tuple of (row_char, *args)
        # row_char is 's', 'r', 'i'
        # in this case, it looks like its all gonna be replace, so we can repr this as 4-tuple
        # tuple of (row1, scale1, row2, scale2)
        #NOTE: Change replace to one less param since 1 is hardcoded at idx 1?

        currCol = currRow = 0

        for _ in range(upper.rows): # stops infinite loops
            if upper.isRef(): break # shortcircuits if needed

            if any(upper.data[i][currCol]!=0 for i in range(currRow, upper.rows)):
                for idx in range(currRow+1,upper.rows):
                    if upper.data[currRow][currCol]==0: continue
                    factor = upper.data[idx][currCol] / upper.data[currRow][currCol]
                    
                    L.append((idx,1,currRow,factor)) # specifically for reversal later; in reality -factor as below lol
                    
                    upper.replace(idx,1,currRow,-factor) # should make everything zero

                currRow +=1 
                
            currCol +=1 #done regardless
        
        for r in L[::-1]:# Going in reverse order to modify L; r stands for replacement
            lower.replace(*r)

        
        return lower, upper
    
    def augment(self, vector):
        """Returns augmented matrix with vector on righthand side"""
        if not isinstance(vector, Vector): raise TypeError(f"{vector} should be a Vector datatype")
        if len(vector) != self.rows: raise ValueError(f"Size of Vector must be same as number of rows!")
        ret = Matrix(self.rows,self.cols+1)
        
        for row in range(self.rows):
            for col in range(self.cols+1):
                if col==self.cols:
                    #add 
                    ret.data[row].append(vector.vals[row])
                else:
                    ret.data[row].append(self.data[row][col])
        
        return ret
    
    def augmentZero(self):
        """Augments a zero column to the end of the Matrix"""
        return self.augment(Vector(*[0 for _ in range(self.rows)]))
    
    
    def NulBasis(self):
        """Returns set of vectors that are basis for null space of self"""
        #TODO: Return vectors corresponding parametric soln to Ax=0
        pass
    
    def NulMatrix(self):#TODO
        """Returns Matrix from the combination of NulBasis Vectors"""
        pass
    
    
    
    def ColMatrix(self):#TODO
        """Returns Matrix from the combination of ColBasis Vectors"""
        return Matrix.colsToMatrix(self.ColBasis())
        
        
    
    def ColBasis(self):
        """Returns set of vectors that are basis for col space of self"""
        rref = self.rref()
        basis = set()
        for row,col in rref.getPivots():
            basis.add(self.getCol(col))
        return basis
        
    
    def RowBasis(self):
        """returns set of vectors that are basis for row space of self"""
        #Same as Col(self^T) ie column space of the transpose of the matrix
        return self.transpose().ColBasis()
    
    
    def inNullSpace(self, vector):
        """Returns True iff vector is in null space of self, ie self * vector = 0 * vector"""
        #TODO
        
        pass
    
    
    def inColSpace(self,vector):
        """Returns True iff vector is in col space of self, ie self.augment(vector) has no pivots in last row"""
        if vector.rows != self.rows: return False
        return self.augment(vector).isConsistent()
        
    
    def rank(self):
        """Returns rank of Matrix - size of basis of col space"""
        return len(self.ColBasis())
    
    def nullity(self):# TODO: test
        """Retursn nullity of Matrix - size of basis of null space"""
        return len(self.NulBasis())
            
    
        
    
class SquareMatrix(Matrix):
    def __init__(self,size,data):
        super().__init__(size,size,data)
        
        
    def minor(self,row,col): 
        """Removes row'th row and col'th column then returns matrix"""
        
        d = [[] for _ in range(self.cols-1)]
        nthRow = 0
        for r in range(self.rows):
            if r==row: continue
            for c in range(self.cols):
                if c==col: continue
                d[nthRow].append(self.data[r][c])
            nthRow +=1
        
            
        return SquareMatrix(self.rows-1,d)
    
    def cofactor(self,row,col):
        """Returns +- the determinant of the minor"""
        sign = 1 if (row+col)%2==0 else -1
        cropped = self.minor(row,col)
        return sign * cropped.det()


                
        
    def det(self): 
        """Determinant of Square matrix, done recursively probably"""
        if self.cols==self.rows==1: return self.data[0][0] # recursive base case
        if self.rows==self.cols==2: # hard coded for case matrix is 2x2 
            return self.data[0][0]*self.data[1][1]-self.data[0][1]*self.data[1][0]

        sum = 0
        col = 0
        for row in range(self.rows):
            sum += self.data[row][col] * self.cofactor(row,col)
                
                
        return sum
        
    
    def adjugate(self):
        """Returns classical adjoint to matrix"""
        d= [[] for _ in range(self.rows)]
        
        for r in range(self.rows):
            for c in range(self.cols):
                sign = 1 if (r+c)%2==0 else -1
                val = sign * self.minor(r,c).det()
                d[r].append(val)
                
                
        return SquareMatrix(self.rows,d).transpose()
        
    
    
    def inv(self): 
        """Returns inverse iff det != 0"""
        determ = self.det()
        if determ == 0: raise ValueError("Matrix is not invertible since determinant is 0!")
        return (1/determ) * self.adjugate()
        
    
    def __pow__(self,exp):
        if not isinstance(exp, int): raise ValueError(f"2nd parameter {exp} must be integer")
        # expand functionality to zero, negative integers once we learn abt inverse matrices
        
        if exp !=1 and self.rows!=self.cols: raise ValueError("Exponentiation only works on square matrices") # remove
        if exp==1: return self
        
        if exp>1:
        
            times = exp - 1
            temp = Matrix(self.rows,self.cols,self.data)
            for _ in range(times):
                temp = temp * self
        
            return temp
        else: #multiply by inverse matrix (1-exp) times
            times = 1 - exp
            inv = inv(self) #TODO, make more effecient
            #TODO: check if invertible first
            temp = Matrix(self.rows,self.cols,self.data)
            for _ in range(times):
                temp = temp * inv
            return temp
        
    def Bcoord(self, vector):# Basis is SquareMatrix
        """Given that self is a Basis of B coordinate system, returns B-coordinates of vector"""
        return self.augment(vector).rref().getCol(self.cols)
    
    def invBcoord(self,vector):# Basis is SquareMatrix
        return self * vector



class IdentityMatrix(SquareMatrix):
    def __init__(self,size):
        data = [[] for _ in range(size)]
        for row in range(size):
            for col in range(size):
                val = 1 if row==col else 0
                data[row].append(val)
        super().__init__(size,data)
        

class ZeroMatrix(Matrix):
    def __init__(self,rows,cols):
        super().__init__(rows,cols,[[0 for i in range(cols)] for j in range(rows)])
        
        
        
class Vector(Matrix): # subclass of Matrix class
    def __init__(self, *args):
        self.vals = list(args)
        super().__init__(len(args),1,[[val] for val in args])
    
    def __str__(self): # needed?
        vals = [round(val,6) for val in self.vals]
        return "<" + ''.join(str(vals))[1:-1] + ">"
    
    def __abs__(self): #for norm 
        """Defaults to 2-norm of vector"""
        return self.norm()

    
    def __repr__(self):
        return str(self)
    
    def __getitem__(self, idx):# TODO: test
        if not (0 <= idx < len(self.vals)): raise ValueError(f"Trying to manipulate {idx}'th entry of Vector {self}")
        return self.vals[idx]
    
    def __setitem__(self,idx, val):# TODO: test
        if not (0 <= idx < len(self.vals)): raise ValueError(f"Trying to manipulate {idx}'th entry of Vector {self}")
        self.vals[idx] = val
        self.data[idx][0] = val


    def __add__(self,other):
        if not isinstance(other,Vector): raise TypeError("Cannot add a vector to a non-vector")
        if len(self)!=len(other): raise ValueError("Cannot add vectors of different sizes")
        return Vector(*(self.vals[i]+other.vals[i] for i in range(len(self))))


    def __sub__(self,other):
        if not isinstance(other,Vector): raise TypeError("Cannot subtract a vector from a non-vector")
        return self + -other

    def __mul__(self,other):
        if isinstance(other,int) or isinstance(other,float): return Vector(*(other*self.vals[i] for i in range(len(self))))
        elif isinstance(other,Vector):
            if len(self)!=len(other): raise ValueError("Cannot find dot product of vectors of different sizes")
            return sum(self.vals[i]*other.vals[i] for i in range(len(self)))
        elif isinstance(other,Matrix):
            mat = Matrix.vectorToMatrix(self)
            return mat * other
        raise TypeError("You attempted to multiply a vector by an invalid datatype :(")
    
    def __rmul__(self,other): 
        if isinstance(other,int) or isinstance(other,float): return self*other # shouldn't be called if both are vectors
        elif isinstance(other,Matrix): 
            mat = Matrix.vectorToMatrix(self)
            return other * mat
        raise TypeError("You attempted to multiply a vector by an invalid dataype :/")
    
    def __neg__(self):
        """Negates vector, ie -1 * self"""
        return Vector(*(-val for val in self))


    def cross(self,other):
        if not isinstance(other,Vector): raise TypeError("A cross product must be between two vectors")
        if len(self) != len(other): raise ValueError("Cannot find cross product between vectors of different sizes")
        if len(self)!=3: raise ValueError("Cannot take cross product on vectors that do not have three dimensions")
        return Vector([self.vals[1]*other.vals[2]-self.vals[2]*other.vals[1], self.vals[2]*other.vals[0]-self.vals[0]*other.vals[2], self.vals[0]*other.vals[1]-self.vals[1]*other.vals[0]])
        # add functionality for 7d and/or 2d?
        
    def normalize(self):
        """Normalizes vector"""
        return (1/abs(self)) * self
    
    def isOrthogonal(self, other):#TODO: test
        """Returns True iff self and other are orthogonal vectors"""
        if not isinstance(other, Vector): raise TypeError(f"{other} is supposed to be a Vector")
        return self * other == 0
    
    def isOrthogonalSet(setOfVectors):#TODO: test
        """Returns True iff the set of vectors are all mutually orthogonal"""
        return all (first.isOrthogonal(second) or first==second for first in setOfVectors for second in setOfVectors)
    
    def distance(self, other):#TODO: test
        """Returns euclidian distance between the tips of two vectors"""
        if not isinstance(other, Vector): raise TypeError(f"{other} is supposed to be a Vector")
        return abs(self - other)
    
    def proj(self, other):
        """Returns the projection of self vector onto other"""
        if not isinstance(other, Vector): raise TypeError(f"{other} is supposed to be a Vector")
        return ((other*self)/(other*other)) * other

    
    def orthogonalProjection(self, other):
        """Decomposes self vector into components parallel and perpendicular to other"""
        if isinstance(other, Vector): 
            yhat = self.proj(other)
            return yhat, (self - yhat)
        elif isinstance(other, set):
            if not all (isinstance(vector, Vector) for vector in other): raise TypeError("Silly goober! Tricks are for kids")
            
            yhat = Vector(*(0 for _ in range(len(self))))
            for vector in other:
                yhat += self.OrthogonalProjection(vector)[0]
            
            z = self - yhat
            
            return yhat, z
            
            
        raise TypeError(f"{other} is supposed to be a Vector")
    
    def isUnitVector(self):#TODO: test
        """Returns True iff self is a unit vector, ie has length 1"""
        return abs(self)==1
    
    def isOrthonormalSet(setOfVectors):#TODO: test
        """Returns True if setOfVectors is orthonormal set"""
        return Vector.isOrthogonalSet(setOfVectors) and all(vector.isUnitVector() for vector in setOfVectors)
    
    def gramSchmidt(setOfVectors):
        """Given input setOfVectors that is a basis for some vector space, outputs a orthogonal basis as per the Gram Schmidt process"""
        if isinstance(setOfVectors,Matrix):
            setOfVectors = setOfVectors.getCols() # TODO: implement this
        if not isinstance(setOfVectors, set) or not isinstance(setOfVectors,list) or not isinstance(setOfVectors,tuple) or not all (isinstance(vector, Vector) for vector in setOfVectors): 
            raise TypeError("Silly goober! Tricks are for kids")

        #NOTE: we are assuming that the input is already a basis for some vector space
        li = list(setOfVectors) #no reason for reverse, just felt like it lol
        ret = [] # return basis of vectors
        
        for i in range(len(li)):
            v = li[i]
            for j in range(i):
                v -=  li[i].proj(ret[j])
            ret.append(v)
            
        return ret

    def norm(self, method=2):
        """
        Returns the norm of the Vector self, of type method
        :param: method - accepts 'f' (Frobenius), any nonnegative integer, or 'inf' (infinity)"""
        # if not ((isinstance(method, int) and method>=0) or (isinstance(method,str) and method=='f') or (isinstance(method, float) and method==float('inf'))):
        #     raise TypeError(f"method parameter {method} must be nonnegative integer, infinity, or f")
        
        match method:
            case 'inf':
                print(self.vals)
                return max(self.vals)
            case 'f':
                return -1 #frobenius norm; extension of 2 norm to matrices
            case '0'|0:
                return sum([1 if val!=0 else 0 for val in self.vals])
            
            case _: 
                return math.pow(sum((abs(val))**method for val in self.vals),(1/method)) 
            

    
    
        
        


"""
The following are equivalent: 
- The matrix A has a pivot in every col
- The system has no free vars
- The system Ax=b has either no solution or one solution depending on b
- Ax=0 has only the trivial solution
- The columns of A are linearly independent
- No column of A can be written as a linear combination of the other cols
- If ax=Ay, then x=y
- The transformation T(x)=Ax is one-to-one ?
- the # of cols is <= # of rows of A (this one follows from others but does not imply the others)

The following are equivalent: 
- The matrix has a col without a pivot
- There is at least one free variable
- Ax=b has either no solution or infinitely many solutions
- Ax=0 has a non-trivial solution
- The cols of A are linearly dependent
- There is a non-trivial linear combination of the cols of A that gives 0
- One col of A can be written as a linear combination of the other cols of A
- The transformation T(x)=Ax is not one-to-one 

### Add to this????????????
"""



if __name__ == '__main__balls':#won't run until I remove balls haha
    
    d1 = [[1,2,3],[0,0,0],[0,0,4]]
    d2 = [[1,2,0],[0,4,5],[0,0,0]]
    d3 = [[0,0,0],[0,0,0],[0,0,0]]
    d4 = [[1,2,3],[0,0,0],[-1,0,1]]
    d5 = [[1,2,3],[0,4,0],[0,6,5]]
    d6 = [[1,0,0],[0,1,0],[0,0,1]]#if we want to
    datas = [d1,d2,d3,d4,d5,d6]
    for d in datas:
        mat = Matrix(3,3,d)
        print(f'Matrix {mat.data} is in REF? {mat.isRef()}')
        print(f'Matrix {mat.data} is in RREF? {mat.isRref()}')
        # should be false, true, true, false, false
        
    m1 = Matrix(2,2,[[1,2],[3,4]])# tests matrix multiplication + exponentiation
    m2 = Matrix(2,2,[[1,2],[3,4]])
    print(m1*m2)#==m1**2 ==7,10,,15,22
    print(m1**3)#==m1*m1*m1 == 37,54,,81,118
    m3 = Matrix(2,3,[[7,6,-2],[3,1,2]])
    m4= Matrix(3,1,[[1],[4],[1]])
    print(m3*m4)
    
    v1 = Vector([1,2,3])
    v2 = Vector([4,5,6])
    print(v1.cross(v2)) # some testing for vectors\
        
    A = Matrix(3,2,[[-1,4],[3,3],[4,-4]])
    B = Matrix(2,2,[[3,-2],[-3,3]])
    print(A*B)
    print(Matrix(2,2,[[4,-6],[-8,12]]) * Matrix(2,2,[[7,-1],[7,1]]))

    I = IdentityMatrix(6)
    print(I)

    Z = ZeroMatrix(5,4)
    print(Z)
    
    M = Matrix(3,2,[[1,2],[3,4],[5,6]])
    print(M.transpose())
    
    m = Matrix(2,2,[[7,2],[-6,-2]])
    v = Matrix(2,1,[[-6],[-4]])
    print(m.inv()*v)
    
    lu1 = Matrix(3,4,[[1,2,0,2],[1,3,2,1],[2,3,4,0]])
    print(lu1.LuFactorization()[0])
    print(lu1.LuFactorization()[1])

    circum = SquareMatrix(3,[[1,2,3],[4,5,6],[7,8,9]])
    print(circum)
    print(circum.minor(2,1))

    deter = SquareMatrix(3,[[1,2,3],[3,2,1],[2,1,3]])
    print(deter)
    print(deter.det())
    
    d = [[-3,-2,2,0],[-5,6,1,0],[6,0,-3,0],[3,8,0,-3]]
    m = SquareMatrix(4,d)

    print(m.ref()[0])
    print(m.det())

    v=  Vector(1,2,3)
    matr = Matrix(3,2,[[0,4],[7,8],[5,6]])
    print(matr.augment(v))
    
    B = SquareMatrix(3,[[1,1,2],[0,-1,-4],[0,0,1]])
    xB = Vector(13,-19,4)
    print(B.Bcoord(xB))
    
    A = Matrix(2,2,[[1,2],[3,4]])
    v = Vector(1,2)
    print(v.rows)
    print(v.cols)
    print(A*v)
  
    M = Matrix(2,3,[[1,2,2],[4,-1,3]])
    print(M.ColBasis())
    print(M.rank())
    
    P = SquareMatrix(2,[[1,2],[3,5]])
    D = SquareMatrix(2,[[1,0],[0,2]])

    print(P*(D**4)*P.inv()) # PDP-1 decomposition, do this out acc, hard part is finding eigenvectors + eigenvalues
    y = Vector(3,2)
    u = Vector(8,4)

    print(Vector(-1/3, -2/3, 1/3).normalize())
    
    s = {Vector(1,1,0,1), Vector(-1,0,1,1), Vector(-1,5,3,-4)}
    x = Vector(4,3,3,-1)
    print(x.orthogonalProjection(s))
    
    s = {Vector(1,-1,-1,1,1), Vector(4,1,6,-6,4), Vector(6,-8,-6,9,1)}
    r = Vector.gramSchmidt(s)
    print(r)
    
    M = Matrix(4,3,[[1,2,4],[2,5,8],[3,8,12],[4,11,16]])
    print(M.allCols())
    print(M.allRows())
    print(M.ColMatrix())


x = Vector(0,.2,1, 2.2)

x1 = Vector(0.7, 0.2, 0.5, 2)
x2 = Vector(0,1,1.5,2.2)
x3 = Vector(0.8,0.1,1.2, 2)
vectors = [x1, x2, x3]
for vector in vectors:
    diff = x-vector
    print(f'0-norm is {diff.norm(method=0)}')
    print(f'1-norm is {diff.norm(method=1)}')
    print(f'2-norm is {diff.norm(method=2)}')
    print(f'infinity-norm is {diff.norm(method="inf")}')