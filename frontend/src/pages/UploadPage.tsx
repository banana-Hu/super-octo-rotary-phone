import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { Steps } from '../components/Layout'
import { useCreation } from '../store'

export function UploadPage() {
  const creation = useCreation(); const navigate = useNavigate(); const [busy, setBusy] = useState(false); const [error, setError] = useState('')
  const previews = useMemo(() => creation.files.map((file) => ({ file, url: URL.createObjectURL(file) })), [creation.files])
  const select = (incoming: FileList | null) => {
    if (!incoming) return
    const next = [...creation.files, ...Array.from(incoming)].slice(0, 20)
    const invalid = next.find((file) => !file.type.startsWith('image/') || file.size > 10 * 1024 * 1024)
    if (invalid) return setError('请选择不超过 10MB 的图片文件。')
    creation.update({ files: next }); setError(incoming.length + creation.files.length > 20 ? '最多选择20张图片。' : '')
  }
  const submit = async () => {
    if (!creation.files.length) return setError('请先选择至少一张图片。')
    setBusy(true); setError('')
    try { const response = await api.upload(creation.files); creation.update({ fileIds: response.file_ids }); navigate('/template') }
    catch (reason) { setError(reason instanceof Error ? reason.message : '上传失败') }
    finally { setBusy(false) }
  }
  return <><Steps current={1}/><div className="page-heading"><h1>先把你的快乐，放进来。</h1><p>一张照片、一段现场，或一句想说的话，都是故事的开始。</p></div><section className="panel"><label className="upload-zone"><strong>点击选择照片</strong><span>从手机相册添加，最多20张</span><input type="file" accept="image/*" multiple onChange={(event) => select(event.target.files)}/></label>{previews.length > 0 && <div className="upload-previews">{previews.map(({ file, url }, index) => <figure key={`${file.name}-${index}`}><img src={url} alt={file.name}/><button type="button" aria-label={`删除${file.name}`} onClick={() => creation.update({ files: creation.files.filter((_, i) => i !== index) })}>×</button></figure>)}</div>}<label className="field-label" htmlFor="description">补充描述（选填）</label><textarea id="description" rows={4} value={creation.description} onChange={(event) => creation.update({ description: event.target.value })} placeholder="描述活动感受，用于风格增强……"/>{error && <p className="error-box" role="alert">{error}</p>}<div className="form-bottom"><LinkButton/><button className="button primary" disabled={busy || !creation.files.length} onClick={submit}>{busy ? '正在上传…' : '下一步：选择模板'}</button></div></section></>
}
function LinkButton(){return <a className="text-link" href="/">← 返回广场</a>}
