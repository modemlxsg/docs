## ArrayBase



### Array

`Array`是一个拥有的数组，直接拥有基础数组元素（就像Vec），它是创建和存储n维数据的默认方式。 `Array <A，D>`有两个类型参数：A表示元素类型，D表示维数。 诸如`Array3 <A>`之类的特定维度的类型别名仅具有元素类型的类型参数A。

```rust
// Create a three-dimensional f64 array, initialized with zeros
use ndarray::Array3;
let mut temperature = Array3::<f64>::zeros((3, 4, 5));
// Increase the temperature in this location
temperature[[2, 2, 2]] += 0.5;
```



**ArcArray**

`ArcArray`是具有引用计数数据（共享所有权）的拥有的数组。 共享要求它使用写时复制进行可变操作。 调用用于使ArcArray上的元素发生变异的方法（例如view_mut（）或get_mut（））将中断共享，并需要克隆数据（如果不是唯一保存的话）。



**CowArray**

CowArray类似于`std :: borrow :: Cow`。 它可以表示不可变的视图或唯一拥有的数组。 如果CowArray实例是不可变的视图变体，则在执行修改之前，调用用于使数组中的元素发生变异的方法将导致其转换为拥有的变体（通过克隆所有元素）。



### Array Views

`ArrayView`和`ArrayViewMut`分别是只读和读写数组视图。 他们使用维数，索引和几乎所有其他方法都与其他数组类型相同。

当特征范围允许时，ArrayBase的方法也适用于数组视图



### Indexing and Dimension

索引语法 : `array[[i,j,...]]`



维度和索引的重要Trait 和Type

- `Dim`值表示维度或索引。 

- `Trait Dimension`由所有维度实现。 它为维和索引定义了许多操作。 

- `Trait IntoDimension`用于转换为Dim值。 

- `Trait ShapeBuilder`是IntoDimension的扩展，在构造数组时使用。 形状不仅描述每个轴的范围，还描述它们的步幅。 

- `Trait NdIndex`是Dimension的扩展，适用于可以与索引语法一起使用的值。



### Loops, Producers and Iterators

`.iter() and .iter_mut()`

这些是数组的元素迭代器，它们按照数组的逻辑顺序产生一个元素序列，这意味着访问元素的顺序将对应于增加最后一个索引first: 0，…0 0;0,…,0,1;0,…0 2，等等。



`.outer_iter() and .axis_iter()`

这些迭代器产生一维较小的数组视图。

例如，对于2D数组，`.outer_iter（）`将产生1D行。 对于3D数组，`.outer_iter（）`生成2D子视图。 `.axis_iter（）`类似于`external_iter（`），但允许您选择要移动的轴。 outside_iter和axis_iter是一维生成器。



`.genrows(), .gencolumns() and .lanes()`

.genrows（）是数组中所有行的生成器（并且是可迭代的）

```rust
use ndarray::Array;

// 1. Loop over the rows of a 2D array
let mut a = Array::zeros((10, 10));
for mut row in a.genrows_mut() {
    row.fill(1.);
}

// 2. Use Zip to pair each row in 2D `a` with elements in 1D `b`
use ndarray::Zip;
let mut b = Array::zeros(a.nrows());

Zip::from(a.genrows())
    .and(&mut b)
    .apply(|a_row, b_elt| {
        *b_elt = a_row[a.ncols() - 1] - a_row[0];
    });
```



### Slicing

可以使用切片来创建数据子集的视图。切片方法包括`.slice（）`、. `slice_mut（）`、`. slice_move（）`和`.slice_collapse（）`。

可以使用宏`s！[]`来传递切片参数，该宏将在所有示例中使用。





























