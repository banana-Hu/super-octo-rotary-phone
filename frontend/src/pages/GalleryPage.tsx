import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type Work } from '../api'
import { Art } from '../components/Art'

export function GalleryPage() {
  const [works, setWorks] = useState<Work[]>([])
  const [error, setError] = useState('')
  useEffect(() => { api.getWorks().then((value) => setWorks(value.items)).catch((reason) => setError(reason.message)) }, [])
  return <><section className="hero"><div className="hero-copy"><p className="eyebrow">FOR EVERY UNFORGETTABLE MOMENT</p><h1>快乐不止当下，<br/>把瞬间带回家。</h1><p className="lead">漫展的心动、演唱会的返场、和朋友的奇遇。<br/>让散落在相册里的热爱，成为独一份的纪念。</p><Link className="button primary" to="/upload">＋ 创作我的记忆</Link></div><div className="hero-poster"><Art title="热爱，就要大声一点！" /></div></section><section><div className="section-heading"><h2>看看别人的快乐切片</h2><span>灵感广场</span></div>{error && <p className="error-box" role="alert">作品加载失败：{error}</p>}<div className="gallery">{works.map((work, index) => <article className="work-card" key={work.work_id}><Art title={['把热爱，装进四格里','今夜，让心跳返场','走过的路，都算数'][index % 3]} kind={work.style === 'anime' ? 'comic' : 'album'}/><div className="card-info"><span className="tag">{work.template} · {work.style}</span><h3>MomentMaker 样例作品</h3><div className="card-meta"><span>{work.nickname}</span><span>♡ {work.likes}</span></div></div></article>)}</div></section></>
}
