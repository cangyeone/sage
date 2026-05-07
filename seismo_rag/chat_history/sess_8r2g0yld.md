# 三维介质波动方程形式是？

- id: `sess_8r2g0yld`
- created_at: 1778169892257
- updated_at: 1778169969768

## user

三维介质波动方程形式是？

## assistant

三维介质中的波动方程根据所描述波的物理性质分为标量波和矢量波两种情况。最常用的两个形式如下：

---

### 1. 标量波动方程（声波、地震P波近似、电磁波标量近似）

\[
\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u
\]

其中：
- \(u(\mathbf{r},t)\) 为波场（如声压、标量位移势等），
- \(c\) 为波速，
- \(\nabla^2\) 为拉普拉斯算子，在笛卡尔坐标下展开为：

\[
\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2}
\]

该方程描述各向同性、均匀介质中无衰减的标量波传播。

---

### 2. 弹性波动方程（Navier 方程，用于固体介质中的矢量波）

对于各向同性、均匀、线弹性介质，位移矢量 \(\mathbf{u}(\mathbf{r},t)\) 满足：

\[
\rho \frac{\partial^2 \mathbf{u}}{\partial t^2} 
= (\lambda + \mu) \nabla (\nabla \cdot \mathbf{u}) 
+ \mu \nabla^2 \mathbf{u} 
+ \mathbf{f}
\]

其中：
- \(\rho\) 为密度，
- \(\lambda, \mu\) 为拉梅常数，
- \(\mathbf{f}\) 为体力（如重力、震源力）。

该方程可分解为：
- **P波（压缩波）**：由标量势 \(\phi\) 描述，满足 \(\displaystyle \frac{\partial^2 \phi}{\partial t^2} = v_p^2 \nabla^2 \phi\)，其中 \(v_p = \sqrt{(\lambda + 2\mu)/\rho}\)。
- **S波（剪切波）**：由矢量势 \(\mathbf{\Psi}\) 描述，满足 \(\displaystyle \frac{\partial^2 \mathbf{\Psi}}{\partial t^2} = v_s^2 \nabla^2 \mathbf{\Psi}\)，其中 \(v_s = \sqrt{\mu/\rho}\)。

---

如果用户需要更具体的方程（如含衰减项、各向异性介质、或频率域形式），可以进一步说明。
