const uploadBtn = document.getElementById('uploadBtn')
const fileInput = document.getElementById('fileInput')
const uploadResult = document.getElementById('uploadResult')
const askBtn = document.getElementById('askBtn')
const questionBox = document.getElementById('question')
const answerDiv = document.getElementById('answer')
let lastFile = null

uploadBtn.onclick = async () => {
  const f = fileInput.files[0]
  if (!f) {
    alert('Choose a file first')
    return
  }
  uploadBtn.disabled = true
  uploadBtn.innerText = 'Uploading...'
  try {
    const fd = new FormData()
    fd.append('file', f)
    const res = await fetch('/upload', { method: 'POST', body: fd })
    if (!res.ok) throw new Error('Upload failed: ' + res.status)
    const data = await res.json()
    uploadResult.innerText = JSON.stringify(data)
    lastFile = data
    // auto-ingest
    const ingRes = await fetch('/ingest', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ file_id: data.file_id, path: data.path }) })
    if (!ingRes.ok) {
      const text = await ingRes.text()
      uploadResult.innerText += '\nIngest failed: ' + text
    } else {
      const ingData = await ingRes.json()
      uploadResult.innerText += '\nIngested: ' + JSON.stringify(ingData)
    }
  } catch (err) {
    uploadResult.innerText = 'Error: ' + err.message
  } finally {
    uploadBtn.disabled = false
    uploadBtn.innerText = 'Upload'
  }
}

askBtn.onclick = async () => {
  const q = questionBox.value
  if (!q) return
  const res = await fetch('/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: q }) })
  const data = await res.json()
  answerDiv.innerText = JSON.stringify(data, null, 2)
}
