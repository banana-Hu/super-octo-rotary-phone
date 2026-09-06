export function Art({ title = '把热爱，装进四格里', kind = 'comic' }: { title?: string; kind?: string }) {
  return <div className={`art art-${kind}`}><div className="art-top">MOMENT ARCHIVE <span>2026 / VOL.01</span></div><div className="art-sun"/><div className="art-orbit"/><div className="art-mountain"/><div className="art-caption">{title}</div><div className="art-bottom">KEEP THIS MOMENT <span>✦</span></div></div>
}
