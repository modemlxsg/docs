# Vue

## 安装

Vue.js 不支持 IE8 及其以下版本，因为 Vue.js 使用了 IE8 不能模拟的 ECMAScript 5 特性

1. 直接`<script>` 引入

2. CDN引入，`https://unpkg.com/vue` 

   ```html
   <script src="https://unpkg.com/vue"></script>
   ```

3. NPM引入

   ```javascript
   npm install vue
   ```

Vue.js 提供一个官方命令行工具 `vue-cli`

```shell
# 全局安装 vue-cli
$ npm install --global vue-cli
# 创建一个基于 webpack 模板的新项目
$ vue init webpack my-project
# 安装依赖，走你
$ cd my-project
$ npm install
$ npm run dev
```

$$

$$



## 起步

### 声明式渲染

Vue.js 的核心是一个允许采用简洁的模板语法来声明式的将数据渲染进 DOM

html

```html
<div id="app">
  {{ message }}
</div>

<div id="app-2">
  <span v-bind:title="message">
    鼠标悬停几秒钟查看此处动态绑定的提示信息！
  </span>
</div>
```

JS

```javascript
var app = new Vue({
  el: '#app',
  data: {
    message: 'Hello Vue!'
  }
})
```



### 条件与循环

html

```html
<!-- v-if -->
<div id="app-3">
  <p v-if="seen">现在你看到我了</p>
</div>
<!-- v-for -->
<div id="app-4">
  <ol>
    <li v-for="todo in todos">
      {{ todo.text }}
    </li>
  </ol>
</div>
```

JS

```javascript
var app3 = new Vue({
  el: '#app-3',
  data: {
    seen: true
  }
})

var app4 = new Vue({
  el: '#app-4',
  data: {
    todos: [
      { text: '学习 JavaScript' },
      { text: '学习 Vue' },
      { text: '整个牛项目' }
    ]
  }
})
```

这个例子演示了我们不仅可以绑定 DOM 文本到数据，也可以绑定 DOM **结构**到数据



### 处理用户输入

html

```html
<div id="app-5">
  <p>{{ message }}</p>
  <button v-on:click="reverseMessage">逆转消息</button>
</div>
```

JS

```javascript
var app5 = new Vue({
  el: '#app-5',
  data: {
    message: 'Hello Vue.js!'
  },
  methods: {
    reverseMessage: function () {
      this.message = this.message.split('').reverse().join('')
    }
  }
})
```

Vue 还提供了 v-model 指令，它能轻松实现表单输入和应用状态之间的双向绑定

```html
<div id="app-6">
  <p>{{ message }}</p>
  <input v-model="message">
</div>
```

```javascript
var app6 = new Vue({
  el: '#app-6',
  data: {
    message: 'Hello Vue!'
  }
})
```

### 组件化应用构建

在 Vue 里，一个组件本质上是一个拥有预定义选项的一个 Vue 实例，在 Vue 中注册组件很简单：

```javascript
// 定义名为 todo-item 的新组件
Vue.component('todo-item', {
  template: '<li>这是个待办项</li>'
})
```

现在你可以用它构建另一个组件模板：

```html
<ol>
  <!-- 创建一个 todo-item 组件的实例 -->
  <todo-item></todo-item>
</ol>
```

但是这样会为每个待办项渲染同样的文本，我们应该能将数据从父作用域传到子组件

```javascript
Vue.component('todo-item', {
  // todo-item 组件现在接受一个
  // "prop"，类似于一个自定义属性
  // 这个属性名为 todo。
  props: ['todo'],
  template: '<li>{{ todo.text }}</li>'
})

var app7 = new Vue({
  el: '#app-7',
  data: {
    groceryList: [
      { id: 0, text: '蔬菜' },
      { id: 1, text: '奶酪' },
      { id: 2, text: '随便其他什么人吃的东西' }
    ]
  }
})
```

现在，我们可以使用 v-bind 指令将待办项传到每一个重复的组件中：

```html
<div id="app-7">
  <ol>
    <!--
      现在我们为每个 todo-item 提供待办项对象
      待办项对象是变量，即其内容可以是动态的。
      我们也需要为每个组件提供一个“key”，晚些时候我们会做个解释。
    -->
    <todo-item
      v-for="item in groceryList"
      v-bind:todo="item"
      v-bind:key="item.id">
    </todo-item>
  </ol>
</div>
```



## Vue实例

### 构造器

每个 Vue.js 应用都是通过构造函数 Vue 创建一个 Vue 的根实例 启动

```javascript
var vm = new Vue({
  // 选项
})
```

在实例化 Vue 时，需要传入一个选项对象，它可以包含数据、模板、挂载元素、方法、生命周期钩子等选项

可以扩展 Vue 构造器，从而用预定义选项创建可复用的组件构造器

```javascript
var MyComponent = Vue.extend({
  // 扩展选项
})
// 所有的 `MyComponent` 实例都将以预定义的扩展选项被创建
var myComponentInstance = new MyComponent()
```

所有的 Vue.js 组件其实都是被扩展的 Vue 实例



### 属性与方法

每个 Vue 实例都会代理其 data 对象里所有的属性

```javascript
var data = { a: 1 }
var vm = new Vue({
  data: data
})
vm.a === data.a // -> true
// 设置属性也会影响到原始数据
vm.a = 2
data.a // -> 2
// ... 反之亦然
data.a = 3
vm.a // -> 3
```

除了 data 属性， Vue 实例暴露了一些有用的实例属性与方法。这些属性与方法都有前缀 $，以便与代理的 data 属性区分

```javascript
var data = { a: 1 }
var vm = new Vue({
  el: '#example',
  data: data
})
vm.$data === data // -> true
vm.$el === document.getElementById('example') // -> true
// $watch 是一个实例方法
vm.$watch('a', function (newVal, oldVal) {
  // 这个回调将在 `vm.a`  改变后调用
})
```



### 实例生命周期

每个 Vue 实例在被创建之前都要经过一系列的初始化过程。例如，实例需要配置数据观测(data observer)、编译模版、挂载实例到 DOM ，然后在数据变化时更新 DOM 。在这个过程中，实例也会调用一些 生命周期钩子 ，这就给我们提供了执行自定义逻辑的机会

```javascript
var vm = new Vue({
  data: {
    a: 1
  },
  created: function () {
    // `this` 指向 vm 实例
    console.log('a is: ' + this.a)
  }
})
// -> "a is: 1"
```

也有一些其它的钩子，在实例生命周期的不同阶段调用，如 `mounted`、`updated`、`destroyed`。钩子的 this 指向调用它的 Vue 实例

![lifecycle](.\img\lifecycle.png)



## 模板语法

### 插值

- 数据绑定最常见的形式就是使用 “Mustache” 语法（双大括号）的文本插值

```html
<span>Message: {{ msg }}</span>

<!--通过使用 v-once 指令，你也能执行一次性地插值，当数据改变时，插值处的内容不会更新-->
<span v-once>This will never change: {{ msg }}</span>
```



- 双大括号会将数据解释为纯文本，而非 HTML 。为了输出真正的 HTML ，你需要使用 v-html 指令

```html
<div v-html="rawHtml"></div>
```

这个 div 的内容将会被替换成为属性值 rawHtml，直接作为 HTML —— 数据绑定会被忽略。注意，你不能使用 v-html 来复合局部模板，因为 Vue 不是基于字符串的模板引擎。组件更适合担任 UI 重用与复合的基本单元



- Mustache 不能在 HTML 属性中使用，应使用 v-bind 指令

```html
<div v-bind:id="dynamicId"></div>
<button v-bind:disabled="isButtonDisabled">Button</button>
```



- 使用 JavaScript 表达式

```html
{{ number + 1 }}
{{ ok ? 'YES' : 'NO' }}
{{ message.split('').reverse().join('') }}
<div v-bind:id="'list-' + id"></div>
```

这些表达式会在所属 Vue 实例的数据作用域下作为 JavaScript 被解析。有个限制就是，每个绑定都只能包含单个表达式

```html
<!-- 这是语句，不是表达式 -->
{{ var a = 1 }}
<!-- 流控制也不会生效，请使用三元表达式 -->
{{ if (ok) { return message } }}
```



### 指令

指令（Directives）是带有 v- 前缀的特殊属性

```html
<!--一些指令能接受一个“参数”，在指令后以冒号指明。-->
<a v-bind:href="url"></a>
```

修饰符（Modifiers）是以半角句号 . 指明的特殊后缀，用于指出一个指令应该以特殊方式绑定

例如，.prevent 修饰符告诉 v-on 指令对于触发的事件调用 event.preventDefault()

```html
<form v-on:submit.prevent="onSubmit"></form>
```





### 缩写

Vue.js 为两个最为常用的指令提供了特别的缩写

`v-bind` 缩写

```html
<!-- 完整语法 -->
<a v-bind:href="url"></a>
<!-- 缩写 -->
<a :href="url"></a>
```

`v-on` 缩写

```html
<!-- 完整语法 -->
<a v-on:click="doSomething"></a>
<!-- 缩写 -->
<a @click="doSomething"></a>
```



## 计算属性

模板内的表达式是非常便利的，但是它们实际上只用于简单的运算

```html
<div id="example">
  <p>Original message: "{{ message }}"</p>
  <p>Computed reversed message: "{{ reversedMessage }}"</p>
</div>
```

```javascript
var vm = new Vue({
  el: '#example',
  data: {
    message: 'Hello'
  },
  computed: {
    // a computed getter
    reversedMessage: function () {
      // `this` points to the vm instance
      return this.message.split('').reverse().join('')
    }
  }
})
```

这里我们声明了一个计算属性 reversedMessage 。我们提供的函数将用作属性 vm.reversedMessage 的 getter

- 计算属性 vs Methods

我们可以通过调用表达式中的 method 来达到同样的效果

```html
<p>Reversed message: "{{ reversedMessage() }}"</p>
// in component
methods: {
  reversedMessage: function () {
    return this.message.split('').reverse().join('')
  }
}
```

然而，不同的是计算属性是基于它们的依赖进行缓存的,这就意味着只要 message 还没有发生改变，多次访问 reversedMessage 计算属性会立即返回之前的计算结果，而不必再次执行函数

- Computed 属性 vs Watched 属性

```html
<div id="demo">{{ fullName }}</div>
```

```javascript
var vm = new Vue({
  el: '#demo',
  data: {
    firstName: 'Foo',
    lastName: 'Bar',
    fullName: 'Foo Bar'
  },
  watch: {
    firstName: function (val) {
      this.fullName = val + ' ' + this.lastName
    },
    lastName: function (val) {
      this.fullName = this.firstName + ' ' + val
    }
  }
})
```

```javascript
var vm = new Vue({
  el: '#demo',
  data: {
    firstName: 'Foo',
    lastName: 'Bar'
  },
  computed: {
    fullName: function () {
      return this.firstName + ' ' + this.lastName
    }
  }
})
```



- 计算属性默认只有 getter ，不过在需要时你也可以提供一个 setter

```javascript
computed: {
  fullName: {
    // getter
    get: function () {
      return this.firstName + ' ' + this.lastName
    },
    // setter
    set: function (newValue) {
      var names = newValue.split(' ')
      this.firstName = names[0]
      this.lastName = names[names.length - 1]
    }
  }
}
```



- 虽然计算属性在大多数情况下更合适，但有时也需要一个自定义的 watcher

当你想要在数据变化响应时，执行异步操作或开销较大的操作，这是很有用的

```html
<div id="watch-example">
  <p>
    Ask a yes/no question:
    <input v-model="question">
  </p>
  <p>{{ answer }}</p>
</div>
```

```javascript
<!-- Since there is already a rich ecosystem of ajax libraries    -->
<!-- and collections of general-purpose utility methods, Vue core -->
<!-- is able to remain small by not reinventing them. This also   -->
<!-- gives you the freedom to just use what you're familiar with. -->
<script src="https://unpkg.com/axios@0.12.0/dist/axios.min.js"></script>
<script src="https://unpkg.com/lodash@4.13.1/lodash.min.js"></script>
<script>
var watchExampleVM = new Vue({
  el: '#watch-example',
  data: {
    question: '',
    answer: 'I cannot give you an answer until you ask a question!'
  },
  watch: {
    // 如果 question 发生改变，这个函数就会运行
    question: function (newQuestion) {
      this.answer = 'Waiting for you to stop typing...'
      this.getAnswer()
    }
  },
  methods: {
    // _.debounce 是一个通过 lodash 限制操作频率的函数。
    // 在这个例子中，我们希望限制访问yesno.wtf/api的频率
    // ajax请求直到用户输入完毕才会发出
    // 学习更多关于 _.debounce function (and its cousin
    // _.throttle), 参考: https://lodash.com/docs#debounce
    getAnswer: _.debounce(
      function () {
        if (this.question.indexOf('?') === -1) {
          this.answer = 'Questions usually contain a question mark. ;-)'
          return
        }
        this.answer = 'Thinking...'
        var vm = this
        axios.get('https://yesno.wtf/api')
          .then(function (response) {
            vm.answer = _.capitalize(response.data.answer)
          })
          .catch(function (error) {
            vm.answer = 'Error! Could not reach the API. ' + error
          })
      },
      // 这是我们为用户停止输入等待的毫秒数
      500
    )
  }
})
</script>
```



## Class 与 Style 绑定

### 绑定 HTML Class

#### 1.对象语法

我们可以传给 v-bind:class 一个对象，以动态地切换 class

```html
<div v-bind:class="{ active: isActive }"></div>
<!-- 上面的语法表示 class active 的更新将取决于数据属性 isActive -->
```

v-bind:class 指令可以与普通的 class 属性共存

```html
<div class="static"
     v-bind:class="{ active: isActive, 'text-danger': hasError }">
</div>
```

```javascript
data: {
  isActive: true,
  hasError: false
}
```

```html
<!-- 渲染结果 -->
<div class="static active"></div>
```

可以直接绑定数据里的一个对象

```html
<div v-bind:class="classObject"></div>
```

```javascript
data: {
  classObject: {
    active: true,
    'text-danger': false
  }
}
```

也可以在这里绑定返回对象的计算属性

```html
<div v-bind:class="classObject"></div>
```

```javascript
data: {
  isActive: true,
  error: null
},
computed: {
  classObject: function () {
    return {
      active: this.isActive && !this.error,
      'text-danger': this.error && this.error.type === 'fatal',
    }
  }
}
```



#### 2.数组语法

我们可以把一个数组传给 v-bind:class ，以应用一个 class 列表

```html
<div v-bind:class="[activeClass, errorClass]">
```

```javascript
data: {
  activeClass: 'active',
  errorClass: 'text-danger'
}
```

当有多个条件 class 时这样写有些繁琐。可以在数组语法中使用对象语法

```html
<div v-bind:class="[{ active: isActive }, errorClass]">
```

#### 3.用在组件上

```javascript
Vue.component('my-component', {
  template: '<p class="foo bar">Hi</p>'
})
```

```html
<my-component class="baz boo"></my-component>
```

```html
<!-- 渲染为 -->
<p class="foo bar baz boo">Hi</p>
```



### 绑定内联样式



```html
<div v-bind:style="{ color: activeColor, fontSize: fontSize + 'px' }"></div>
```

```javascript
data: {
  activeColor: 'red',
  fontSize: 30
}
```

同样可以使用对象、数组语法



## 条件渲染

### v-if

```html
<h1 v-if="ok">Yes</h1>
<h1 v-else>No</h1>
```

因为 `v-if` 是一个指令，需要将它添加到一个元素上。但是如果我们想切换多个元素呢？此时我们可以把一个 `<template>` 元素当做包装元素，并在上面使用 v-if

```html
<template v-if="ok">
  <h1>Title</h1>
  <p>Paragraph 1</p>
  <p>Paragraph 2</p>
</template>
```

`v-else-if`，顾名思义，充当 v-if 的“else-if 块”。可以链式地使用多次

```html
<div v-if="type === 'A'">
  A
</div>
<div v-else-if="type === 'B'">
  B
</div>
<div v-else-if="type === 'C'">
  C
</div>
<div v-else>
  Not A/B/C
</div>
```

**用 key 管理可复用的元素**

Vue 会尽可能高效地渲染元素，通常会复用已有元素而不是从头开始渲染

```html
<template v-if="loginType === 'username'">
  <label>Username</label>
  <input placeholder="Enter your username">
</template>
<template v-else>
  <label>Email</label>
  <input placeholder="Enter your email address">
</template>
```

上面的代码中切换 loginType 将不会清除用户已经输入的内容。因为两个模版使用了相同的元素

```html
<template v-if="loginType === 'username'">
  <label>Username</label>
  <input placeholder="Enter your username" key="username-input">
</template>
<template v-else>
  <label>Email</label>
  <input placeholder="Enter your email address" key="email-input">
</template>
```

现在，每次切换时，输入框都将被重新渲染



### v-show

另一个用于根据条件展示元素的选项是 v-show 指令。用法大致一样。v-show 不支持 <template> 语法，也不支持 v-else

不同的是带有 v-show 的元素始终会被渲染并保留在 DOM 中。v-show 是简单地切换元素的 CSS 属性 display



## 列表渲染

我们用 `v-for` 指令根据一组数组的选项列表进行渲染。 

`v-for` 指令需要以 `item in items` 形式的特殊语法， `items` 是源数据数组并且 `item` 是数组元素迭代的别名。

### 基本用法

```html
<ul id="example-1">
  <li v-for="item in items">
    {{ item.message }}
  </li>
</ul>
```

```javascript
var example1 = new Vue({
  el: '#example-1',
  data: {
    items: [
      {message: 'Foo' },
      {message: 'Bar' }
    ]
  }
})
```

在 v-for 块中，我们拥有对父作用域属性的完全访问权限。 v-for 还支持一个可选的第二个参数为当前项的索引

```html
<ul id="example-2">
  <li v-for="(item, index) in items">
    {{ parentMessage }} - {{ index }} - {{ item.message }}
  </li>
</ul>
```

```javascript
var example2 = new Vue({
  el: '#example-2',
  data: {
    parentMessage: 'Parent',
    items: [
      { message: 'Foo' },
      { message: 'Bar' }
    ]
  }
})
```

结果为

- Parent-0-Foo
- Parent-1-Bar



### Template `v-for`

如同 v-if 模板，你也可以用带有 v-for 的 <template> 标签来渲染多个元素块

```html
<ul>
  <template v-for="item in items">
    <li>{{ item.msg }}</li>
    <li class="divider"></li>
  </template>
</ul>
```



### 对象迭代 `v-for`

也可以用 v-for 通过一个对象的属性来迭代

```html
<ul id="repeat-object" class="demo">
  <li v-for="value in object">
    {{ value }}
  </li>
</ul>
```

```javascript
new Vue({
  el: '#repeat-object',
  data: {
    object: {
      firstName: 'John',
      lastName: 'Doe',
      age: 30
    }
  }
})
```

结果为

- John
- Doe
- 30

也可以提供第二个的参数为键名

```html
<div v-for="(value, key) in object">
  {{ key }} : {{ value }}
</div>
```

第三个参数为索引：

```html
<div v-for="(value, key, index) in object">
  {{ index }}. {{ key }} : {{ value }}
</div>
```



### 整数迭代`v-for`

```html
<div>
  <span v-for="n in 10">{{ n }} </span>
</div>
```



### 组件和`v-for`

在自定义组件里，你可以像任何普通元素一样用 v-for

```html
<my-component v-for="item in items" :key="item.id"></my-component>
```

2.2.0+ 的版本里，当在组件中使用 v-for 时，key 现在是必须的

然而他不能自动传递数据到组件里，因为组件有自己独立的作用域。为了传递迭代数据到组件里，我们要用 props

```html
<my-component
  v-for="(item, index) in items"
  v-bind:item="item"
  v-bind:index="index"
  v-bind:key="item.id"
></my-component>
```



### `v-for` with `v-if`

当它们处于同一节点， v-for 的优先级比 v-if 更高，这意味着 v-if 将分别重复运行于每个 v-for 循环中

```html
<li v-for="todo in todos" v-if="!todo.isComplete">
  {{ todo }}
</li>
```



### 列表更新检测

Vue 包含一组观察数组的变异方法（改变被这些方法调用的原始数组），所以它们也将会触发视图更新。这些方法如下：

- push()
- pop()
- shift()
- unshift()
- splice()
- sort()
- reverse()

```javascript
example1.items.push({ message: 'Baz' })
```

非变异方法（non-mutating method）
```javascript
example1.items = example1.items.filter(function (item) {
  return item.message.match(/Foo/)
})
```

>注意事项：
>
>由于 JavaScript 的限制， Vue 不能检测以下变动的数组：
>当你利用索引直接设置一个项时，例如： vm.items[indexOfItem] = newValue
>当你修改数组的长度时，例如： vm.items.length = newLength
>
>为了解决第一类问题，以下两种方式都可以实现
>
>Vue.set(example1.items, indexOfItem, newValue)
>
>为了解决第二类问题，你可以使用 splice
>
>example1.items.splice(newLength)



### 显示 过滤/排序 结果

有时，我们想要显示一个数组的过滤或排序副本，而不实际改变或重置原始数据。在这种情况下，可以创建返回过滤或排序数组的计算属性

```html
<li v-for="n in evenNumbers">{{ n }}</li>
```

```javascript
data: {
  numbers: [ 1, 2, 3, 4, 5 ]
},
computed: {
  evenNumbers: function () {
    return this.numbers.filter(function (number) {
      return number % 2 === 0
    })
  }
}
```

在计算属性不适用的情况下 (例如，在嵌套 v-for 循环中) 你可以使用一个 method 方法

```html
<li v-for="n in even(numbers)">{{ n }}</li>
```

```javascript
data: {
  numbers: [ 1, 2, 3, 4, 5 ]
},
methods: {
  even: function (numbers) {
    return numbers.filter(function (number) {
      return number % 2 === 0
    })
  }
}
```



## 事件处理器

### 监听事件

可以用 v-on 指令监听 DOM 事件来触发一些 JavaScript 代码

```html
<div id="example-1">
  <button v-on:click="counter += 1">增加 1</button>
  <p>这个按钮被点击了 {{ counter }} 次。</p>
</div>
```

```javascript
var example1 = new Vue({
  el: '#example-1',
  data: {
    counter: 0
  }
})
```



### 方法事件处理器

许多事件处理的逻辑都很复杂，所以直接把 JavaScript 代码写在 v-on 指令中是不可行的

```html
<div id="example-2">
  <!-- `greet` 是在下面定义的方法名 -->
  <button v-on:click="greet">Greet</button>
</div>
```

```javascript
var example2 = new Vue({
  el: '#example-2',
  data: {
    name: 'Vue.js'
  },
  // 在 `methods` 对象中定义方法
  methods: {
    greet: function (event) {
      // `this` 在方法里指当前 Vue 实例
      alert('Hello ' + this.name + '!')
      // `event` 是原生 DOM 事件
      if (event) {
        alert(event.target.tagName)
      }
    }
  }
})
// 也可以用 JavaScript 直接调用方法
example2.greet() // -> 'Hello Vue.js!'
```



### 事件修饰符

Vue.js 为 v-on 提供了 事件修饰符。通过由点(.)表示的指令后缀来调用修饰符。

- .stop
- .prevent
- .capture
- .self
- .once

```html
<!-- 阻止单击事件冒泡 -->
<a v-on:click.stop="doThis"></a>
<!-- 提交事件不再重载页面 -->
<form v-on:submit.prevent="onSubmit"></form>
<!-- 修饰符可以串联  -->
<a v-on:click.stop.prevent="doThat"></a>
<!-- 只有修饰符 -->
<form v-on:submit.prevent></form>
<!-- 添加事件侦听器时使用事件捕获模式 -->
<div v-on:click.capture="doThis">...</div>
<!-- 只当事件在该元素本身（比如不是子元素）触发时触发回调 -->
<div v-on:click.self="doThat">...</div>
```



### 键值修饰符

在监听键盘事件时，我们经常需要监测常见的键值。 Vue 允许为 v-on 在监听键盘事件时添加关键修饰符：

```html
<!-- 只有在 keyCode 是 13 时调用 vm.submit() -->
<input v-on:keyup.13="submit">
```

记住所有的 keyCode 比较困难，所以 Vue 为最常用的按键提供了别名

```javascript
<!-- 同上 -->
<input v-on:keyup.enter="submit">
<!-- 缩写语法 -->
<input @keyup.enter="submit">
```

全部的按键别名：

- .enter
- .tab
- .delete (捕获 “删除” 和 “退格” 键)
- .esc
- .space
- .up
- .down
- .left
- .right



### 修饰键

可以用如下修饰符开启鼠标或键盘事件监听，使在按键按下时发生响应。

- .ctrl
- .alt
- .shift
- .meta（windows系统win键）
- .left (以下鼠标)
- .right
- .middle

```html
<!-- Alt + C -->
<input @keyup.alt.67="clear">
<!-- Ctrl + Click -->
<div @click.ctrl="doSomething">Do something</div>
```



## 表单控件绑定

可以用 `v-model` 指令在表单控件元素上创建双向数据绑定。它会根据控件类型自动选取正确的方法来更新元素

### 基础用法

#### 文本

```html
<input v-model="message" placeholder="edit me">
<p>Message is: {{ message }}</p>
```

#### 多行文本

在文本区域插值( <textarea>{{ message }}</textarea> ) 并不会生效，应用 v-model 来代替

```html
<span>Multiline message is:</span>
<p style="white-space: pre-line">{{ message }}</p>
<br>
<textarea v-model="message" placeholder="add multiple lines"></textarea>
```

#### 复选框

```html
<!-- 单个复选框 -->
<input type="checkbox" id="checkbox" v-model="checked">
<label for="checkbox">{{ checked }}</label>
```

```html
<!-- 多个复选框 -->
<input type="checkbox" id="jack" value="Jack" v-model="checkedNames">
<label for="jack">Jack</label>
<input type="checkbox" id="john" value="John" v-model="checkedNames">
<label for="john">John</label>
<input type="checkbox" id="mike" value="Mike" v-model="checkedNames">
<label for="mike">Mike</label>
<br>
<span>Checked names: {{ checkedNames }}</span>
```

Checked names: [ "Jack", "John", "Mike" ]

#### 单选按钮

```html
<div id="example-4">
  <input type="radio" id="one" value="One" v-model="picked">
  <label for="one">One</label>
  <br>
  <input type="radio" id="two" value="Two" v-model="picked">
  <label for="two">Two</label>
  <br>
  <span>Picked: {{ picked }}</span>
</div>
```

```javascript
new Vue({
  el: '#example-4',
  data: {
    picked: ''
  }
})
```

#### 选择列表

单选列表

```javascript
<div id="example-5">
  <select v-model="selected">
    <option disabled value="">请选择</option>
    <option>A</option>
    <option>B</option>
    <option>C</option>
  </select>
  <span>Selected: {{ selected }}</span>
</div>
```

```javascript
new Vue({
  el: '...',
  data: {
    selected: ''
  }
})
```

多选列表（绑定到一个数组）

```html
<div id="example-6">
  <select v-model="selected" multiple style="width: 50px">
    <option>A</option>
    <option>B</option>
    <option>C</option>
  </select>
  <br>
  <span>Selected: {{ selected }}</span>
</div>
```

```javascript
new Vue({
  el: '#example-6',
  data: {
    selected: []
  }
})
```

动态选项，用 v-for 渲染：

```html
<select v-model="selected">
  <option v-for="option in options" v-bind:value="option.value">
    {{ option.text }}
  </option>
</select>
<span>Selected: {{ selected }}</span>
```

```javascript
new Vue({
  el: '...',
  data: {
    selected: 'A',
    options: [
      { text: 'One', value: 'A' },
      { text: 'Two', value: 'B' },
      { text: 'Three', value: 'C' }
    ]
  }
})
```



### 绑定 value

对于单选按钮，勾选框及选择列表选项， v-model 绑定的 value 通常是静态字符串（对于勾选框是逻辑值）

```html
<!-- 当选中时，`picked` 为字符串 "a" -->
<input type="radio" v-model="picked" value="a">
<!-- `toggle` 为 true 或 false -->
<input type="checkbox" v-model="toggle">
<!-- 当选中时，`selected` 为字符串 "abc" -->
<select v-model="selected">
  <option value="abc">ABC</option>
</select>
```

但是有时我们想绑定 value 到 Vue 实例的一个动态属性上，这时可以用 v-bind 实现，并且这个属性的值可以不是字符串

复选框

```html
<input
  type="checkbox"
  v-model="toggle"
  v-bind:true-value="a"
  v-bind:false-value="b"
>
```

```javascript
// 当选中时
vm.toggle === vm.a
// 当没有选中时
vm.toggle === vm.b
```

单选按钮

```html
<input type="radio" v-model="pick" v-bind:value="a">
```

```javascript
// 当选中时
vm.pick === vm.a
```

选择列表设置

```html
<select v-model="selected">
    <!-- 内联对象字面量 -->
  <option v-bind:value="{ number: 123 }">123</option>
</select>
```

```javascript
// 当选中时
typeof vm.selected // -> 'object'
vm.selected.number // -> 123
```



### 修饰符

#### .lazy

在默认情况下， v-model 在 input 事件中同步输入框的值与数据。但你可以添加一个修饰符 lazy ，从而转变为在 change 事件中同步

```html
<!-- 在 "change" 而不是 "input" 事件中更新 -->
<input v-model.lazy="msg" >
```

#### .number

如果想自动将用户的输入值转为 Number 类型（如果原值的转换结果为 NaN 则返回原值），可以添加一个修饰符 number 给 v-model 来处理输入值

```html
<input v-model.number="age" type="number">
```

这通常很有用，因为在 type="number" 时 HTML 中输入的值也总是会返回字符串类型

#### .trim

如果要自动过滤用户输入的首尾空格，可以添加 trim 修饰符到 v-model 上过滤输入：

```html
<input v-model.trim="msg">
```



## 组件

组件 (Component) 是 Vue.js 最强大的功能之一。组件可以扩展 HTML 元素，封装可重用的代码。在较高层面上，组件是自定义元素，Vue.js 的编译器为它添加特殊功能。在有些情况下，组件也可以是原生 HTML 元素的形式，以 is 特性扩展。

### 注册

```
new Vue({
  el: '#some-element',
  // 选项
})
```

```javascript
// 全局注册 
Vue.component('my-component', {
  // 选项
})
```

对于自定义标签名，Vue.js 不强制要求遵循 W3C 规则 (小写，并且包含一个短杠)，尽管遵循这个规则比较好

```javascript
//局部注册，使组件仅在另一个实例/组件的作用域中可用
var Child = {
  template: '<div>A custom component!</div>'
}
new Vue({
  // ...
  components: {
    // <my-component> 将只在父模板可用
    'my-component': Child
  }
})
```

这种封装也适用于其它可注册的 Vue 功能，如指令



### DOM 模版解析说明

当使用 DOM 作为模版时 (例如，将 el 选项挂载到一个已存在的元素上), 你会受到 HTML 的一些限制，因为 Vue 只有在浏览器解析和标准化 HTML 后才能获取模版内容。尤其像这些元素 <ul>，<ol>，<table>，<select> 限制了能被它包裹的元素，而一些像 <option> 这样的元素只能出现在某些其它元素内部

在自定义组件中使用这些受限制的元素时会导致一些问题，例如

```html
<table>
  <my-row>...</my-row>
</table>
```

自定义组件 <my-row> 被认为是无效的内容，因此在渲染的时候会导致错误。变通的方案是使用特殊的 is 属性：

```html
<table>
  <tr is="my-row"></tr>
</table>
```



### 组件通信

**组件 A 在它的模版中使用了组件 B。它们之间必然需要相互通信：父组件要给子组件传递数据，子组件需要将它内部发生的事情告知给父组件**

**在 Vue 中，父子组件的关系可以总结为 props down, events up。父组件通过 props 向下传递数据给子组件，子组件通过 events 给父组件发送消息**



### Prop

#### 使用 Prop 传递数据

组件实例的作用域是孤立的。这意味着不能 (也不应该) 在子组件的模板内直接引用父组件的数据。要让子组件使用父组件的数据，我们需要通过子组件的 props 选项。
子组件要显式地用 props 选项声明它期待获得的数据

```javascript
Vue.component('child', {
  // 声明 props
  props: ['message'],
  // 就像 data 一样，prop 可以用在模板内
  // 同样也可以在 vm 实例中像“this.message”这样使用
  template: '<span>{{ message }}</span>'
})
```

```html
<child message="hello!"></child>
```



#### camelCased (驼峰式)  VS  kebab-case (短横线隔开式)

HTML 特性是不区分大小写的。所以，当使用的不是字符串模版，camelCased (驼峰式) 命名的 prop 需要转换为相对应的 kebab-case (短横线隔开式) 命名

```javascript
Vue.component('child', {
  // camelCase in JavaScript
  props: ['myMessage'],
  template: '<span>{{ myMessage }}</span>'
})
```

```html
<!-- kebab-case in HTML -->
<child my-message="hello!"></child>
```



#### 动态Prop

在模板中，要动态地绑定父组件的数据到子模板的 props，与绑定到任何普通的HTML特性相类似，就是用 v-bind

```html
<div>
  <input v-model="parentMsg">
  <br>
  <child v-bind:my-message="parentMsg"></child>
</div>
```



#### 单向数据流

prop 是单向绑定的：当父组件的属性变化时，将传导给子组件，但是不会反过来。这意味着你不应该在子组件内部改变 prop。

注意在 JavaScript 中对象和数组是引用类型，指向同一个内存空间，如果 prop 是一个对象或数组，在子组件内部改变它会影响父组件的状态。



#### Prop验证

我们可以为组件的 props 指定验证规格。如果传入的数据不符合规格，Vue 会发出警告。当组件给其他人使用时，这很有用。

```javascript
Vue.component('example', {
  props: {
    // 基础类型检测 (`null` 意思是任何类型都可以)
    propA: Number,
    // 多种类型
    propB: [String, Number],
    // 必传且是字符串
    propC: {
      type: String,
      required: true
    },
    // 数字，有默认值
    propD: {
      type: Number,
      default: 100
    },
    // 数组/对象的默认值应当由一个工厂函数返回
    propE: {
      type: Object,
      default: function () {
        return { message: 'hello' }
      }
    },
    // 自定义验证函数
    propF: {
      validator: function (value) {
        return value > 10
      }
    }
  }
})
```



#### 非Prop属性

所谓非 prop 属性，就是它可以直接传入组件，而不需要定义相应的 prop

明确给组件定义 prop 是传参的推荐方式，但组件的作者并不总能预见到组件被使用的场景。所以，组件可以接收任意传入的属性，这些属性都会被添加到组件的根元素上



### 自定义事件

我们知道，父组件是使用 props 传递数据给子组件，但如果子组件要把数据传递回去，应该怎样做？那就是自定义事件

#### 使用 v-on 绑定自定义事件

每个 Vue 实例都实现了事件接口 (Events interface)，即：
使用 `$on(eventName)` 监听事件
使用 `$emit(eventName)` 触发事件

```html
<div id="counter-event-example">
  <p>{{ total }}</p>
  <button-counter v-on:increment="incrementTotal"></button-counter>
  <button-counter v-on:increment="incrementTotal"></button-counter>
</div>
```

```javascript
Vue.component('button-counter', {
  template: '<button v-on:click="incrementCounter">{{ counter }}</button>',
  data: function () {
    return {
      counter: 0
    }
  },
  methods: {
    incrementCounter: function () {
      this.counter += 1
      this.$emit('increment')
    }
  },
})
new Vue({
  el: '#counter-event-example',
  data: {
    total: 0
  },
  methods: {
    incrementTotal: function () {
      this.total += 1
    }
  }
})
```



### 给组件绑定原生事件

有时候，你可能想在某个组件的根元素上监听一个原生事件。可以使用 .native 修饰 v-on

```html
<my-component v-on:click.native="doTheThing"></my-component>
```



### 非父子组件通信

有时候两个组件也需要通信 (非父子关系)。在简单的场景下，可以使用一个空的 Vue 实例作为中央事件总线：

```javascript
var bus = new Vue()
```

```javascript
// 触发组件 A 中的事件
bus.$emit('id-selected', 1)
```

```javascript
bus.$on('id-selected', function (id) {
  // ...
})
```



### 使用Slot分发内容

```js
<alert-box>
  Something bad happened.
</alert-box>

Vue.component('alert-box', {
  template: `
    <div class="demo-alert-box">
      <strong>Error!</strong>
      <slot></slot>
    </div>
  `
})
```



