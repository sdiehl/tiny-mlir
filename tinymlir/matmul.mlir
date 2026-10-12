// Each block stages a TILE x TILE slice of A and B in shared memory per step along K,
// so every global element is loaded once per block rather than once per thread.
func.func private @$name(%a: $view, %b: $view, %out: $view) {
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %i2 = arith.constant 2 : index
  %tile = arith.constant $tile : index
  %zero = arith.constant 0.0 : $t
  %m = memref.dim %a, %i$row : $view
  %k = memref.dim %a, %i$col : $view
  %n = memref.dim %b, %i$col : $view
  %batch = memref.dim %a, %i0 : $view
  %gx = arith.ceildivui %n, %tile : index
  %gy = arith.ceildivui %m, %tile : index
  gpu.launch blocks(%bx, %by, %bz) in (%sx = %gx, %sy = %gy, %sz = $batch)
             threads(%tx, %ty, %tz) in (%ux = %tile, %uy = %tile, %uz = %i1)
             workgroup(%sa : $shared, %sb : $shared) {
    %r = arith.muli %by, %tile : index
    %row = arith.addi %r, %ty : index
    %c = arith.muli %bx, %tile : index
    %col = arith.addi %c, %tx : index
    %rowok = arith.cmpi ult, %row, %m : index
    %colok = arith.cmpi ult, %col, %n : index
    %sum = scf.for %k0 = %i0 to %k step %tile iter_args(%acc = %zero) -> ($t) {
      %ka = arith.addi %k0, %tx : index
      %kaok = arith.cmpi ult, %ka, %k : index
      %aok = arith.andi %rowok, %kaok : i1
      %av = scf.if %aok -> ($t) {
        %v = memref.load %a[$z%row, %ka] : $view
        scf.yield %v : $t
      } else {
        scf.yield %zero : $t
      }
      memref.store %av, %sa[%ty, %tx] : $shared
      %kb = arith.addi %k0, %ty : index
      %kbok = arith.cmpi ult, %kb, %k : index
      %bok = arith.andi %kbok, %colok : i1
      %bv = scf.if %bok -> ($t) {
        %v = memref.load %b[$z%kb, %col] : $view
        scf.yield %v : $t
      } else {
        scf.yield %zero : $t
      }
      memref.store %bv, %sb[%ty, %tx] : $shared
      gpu.barrier
      %part = scf.for %kk = %i0 to %tile step %i1 iter_args(%s = %acc) -> ($t) {
        %x = memref.load %sa[%ty, %kk] : $shared
        %y = memref.load %sb[%kk, %tx] : $shared
        %p = arith.mulf %x, %y : $t
        %next = arith.addf %s, %p : $t
        scf.yield %next : $t
      }
      gpu.barrier
      scf.yield %part : $t
    }
    %ok = arith.andi %rowok, %colok : i1
    scf.if %ok {
      %old = memref.load %out[$z%row, %col] : $view
      %new = arith.addf %old, %sum : $t
      memref.store %new, %out[$z%row, %col] : $view
    }
    gpu.terminator
  }
  return
}
