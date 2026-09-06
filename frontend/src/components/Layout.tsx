import { NavLink, Outlet } from 'react-router-dom'
import { isMockMode } from '../api'

export function Layout() {
  return <><header className="header"><NavLink className="brand" to="/">MomentMaker<i>瞬间制造所</i></NavLink><nav><NavLink to="/">发现广场</NavLink><NavLink to="/upload">开始创作</NavLink></nav><NavLink className="header-cta" to="/upload">创造一个瞬间 ↗</NavLink></header>{isMockMode && <div className="mode-banner">演示模式 · 当前使用本地样例数据</div>}<main><Outlet /></main><footer><span className="brand">MomentMaker ✦</span><span>把你的故事，变成传得出去的记忆。</span></footer></>
}

export function Steps({ current }: { current: number }) {
  return <div className="steps">{['添加素材', '选择模板', '微调画面', '预览成品'].map((label, index) => <span key={label} className={index + 1 === current ? 'current' : ''}><b>0{index + 1}</b>{label}</span>)}</div>
}
